import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import lepm.lattice_elasticity as le
import lepm.lattice_class as lattice_class
import lepm.lattice_functions as latfns
import lepm.gyro_lattice_functions as glatfns
from gyro_lattice_class import GyroLattice
import lepm.plotting.plotting as leplt
import lepm.dataio as dio
import lepm.plotting.colormaps as lecmaps
import lepm.plotting.movies as lemov
import lepm.gyro_data_handling as gdh
import glob
import subprocess
import sys
import lepm.plotting.colormaps as lecmaps
import lepm.stringformat as sf
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

'''Auxiliary script-like functions called by gyro_lattice_class.py when __name__=='__main__'.
'''


def dispersion_abtransition(lp, invert_bg=False, color1=None, color2=None, color_thres=None,
                            nkxvals=50, dab_step=0.005, dpi=100, plot_positive_only=True, ab_max=2.6):
    """

    Parameters
    ----------
    lp

    Returns
    -------

    """
    if invert_bg:
        plt.style.use('dark_background')

    if invert_bg:
        if color1 is None:
            # blue
            color1 = '#70a6ff'
        if color2 is None:
            # red
            color2 = '#ff7777'  # '#90354c'
        # print 'color1, 2 = ', color1, color2
        # sys.exit()
    else:
        cmap = lecmaps.diverging_cmap(250, 10, l=30)
        color1 = cmap(0.)
        color2 = cmap(1.)

    meshfn = le.find_meshfn(lp)
    lp['meshfn'] = meshfn
    lpmaster = copy.deepcopy(lp)
    ablist = np.arange(0, ab_max + dab_step, dab_step)
    # create a place to put images
    outdir = dio.prepdir(meshfn) + 'dispersion_abtransition/'
    dio.ensure_dir(outdir)
    fs = 20

    # go through each ab and make the image
    ii = 0
    for ab in ablist:
        print 'ab = ', ab
        lp = copy.deepcopy(lpmaster)
        lp['ABDelta'] = ab
        lat = lattice_class.Lattice(lp)
        lat.load(check=lp['check'])

        glat = GyroLattice(lat, lp)
        # glat.get_eigval_eigvect(attribute=True)
        fig, ax = leplt.initialize_portrait(ax_pos=[0.12, 0.2, 0.76, 0.6])
        omegas, kx, ky = glat.infinite_dispersion(save=False, nkxvals=nkxvals, nkyvals=50, outdir=outdir, save_plot=False)

        # plot and save it
        title = r'$\Delta_{AB} =$' + '{0:0.2f}'.format(ab)
        for jj in range(len(ky)):
            for kk in range(len(omegas[0, jj, :])):
                if color_thres is None or ab > 0.26:
                    if invert_bg:
                        ax.plot(kx, omegas[:, jj, kk], 'w-', lw=max(1., 30. / (len(kx) * len(ky))))
                    else:
                        ax.plot(kx, omegas[:, jj, kk], 'k-', lw=max(1., 30. / (len(kx) * len(ky))))
                else:
                    # if positive frequencies, color top band blue
                    if len(np.where(omegas[:, jj, kk] > 0)[0]) > 0.5 * len(omegas[:, jj, kk]):
                        # color top band blue
                        if len(np.where(np.abs(omegas[:, jj, kk]) > color_thres)[0]) > 0.5 * len(omegas[:, jj, kk]):
                            ax.plot(kx, omegas[:, jj, kk], '-', lw=max(1., 30. / (len(kx) * len(ky))), color=color1)
                        else:
                            ax.plot(kx, omegas[:, jj, kk], '-', lw=max(1., 30. / (len(kx) * len(ky))), color=color2)

                    # if negative frequencies, color bottom band blue
                    if len(np.where(omegas[:, jj, kk] < 0)[0]) > 0.5 * len(omegas[:, jj, kk]):
                        # color bottom band blue
                        if len(np.where(np.abs(omegas[:, jj, kk]) > color_thres)[0]) > 0.5 * len(omegas[:, jj, kk]):
                            ax.plot(kx, omegas[:, jj, kk], '-', lw=max(1., 30. / (len(kx) * len(ky))), color=color2)
                        else:
                            ax.plot(kx, omegas[:, jj, kk], '-', lw=max(1., 30. / (len(kx) * len(ky))), color=color1)

        ax.text(0.5, 1.15, title, fontsize=fs, ha='center', va='center', transform=ax.transAxes)
        ax.set_xlim(-np.pi, np.pi)
        ax.xaxis.set_ticks([-np.pi, 0, np.pi])
        ax.xaxis.set_ticklabels([r'$-\pi$', 0, r'$\pi$'])
        ax.set_xlabel(r'$k_x$', fontsize=fs)
        ax.set_ylabel(r'$\omega$', fontsize=fs)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fs)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fs)
        if ii == 0:
            ylims = ax.get_ylim()

        if plot_positive_only:
            ax.set_ylim(0., ylims[1])
        else:
            ax.set_ylim(ylims[0], ylims[1])

        # Save it
        name = outdir + 'dispersion{0:04d}'.format(ii)
        plt.savefig(name + '.png', dpi=dpi)
        plt.close('all')

        ii += 1

    # Turn images into a movie
    imgname = outdir + 'dispersion'
    movname = dio.prepdir(meshfn) + 'dispersion_abtrans' + glat.lp['meshfn_exten']
    lemov.make_movie(imgname, movname, indexsz='04', framerate=10, imgdir=outdir, rm_images=True,
                     save_into_subdir=True)


def dice_glat(lp, args):
    """make the OmK arrays to load when computing cherns for diced networks
        Example usage:
        python run_series.py -pro gyro_lattice_class -opts LT/hexagonal/-N/11/-shape/square/-dice_glat/-gridspacing/3.0 -var weakbond_val n1.0:0.1:0.05

    Parameters
    ----------
    lp
    args

    Returns
    -------

    """
    import lepm.line_segments as lsegs
    # Create a glat with bonds that are weakened in a gridlike fashion (bonds crossing gridlines are weak)
    meshfn = le.find_meshfn(lp)
    lp['meshfn'] = meshfn
    lat = lattice_class.Lattice(lp)
    lat.load()

    # Find where bonds cross gridlines
    gridspacing = args.gridspacing
    lp['OmKspec'] = 'gridlines{0:0.2f}'.format(gridspacing).replace('.', 'p') + \
                    'strong{0:0.3f}'.format(lp['Omk']).replace('.', 'p').replace('-', 'n') + \
                    'weak{0:0.3f}'.format(args.weakbond_val).replace('.', 'p').replace('-', 'n')
    maxval = max(np.max(np.abs(lat.xy[:, 0])), np.max(np.abs(lat.xy[:, 1]))) + 1
    gridright = np.arange(gridspacing, maxval, gridspacing)
    gridleft = -gridright
    gridvals = np.hstack((gridleft, 0, gridright))
    # Draw grid
    gridlinesH = np.array([[-maxval, gridv, maxval, gridv] for gridv in gridvals])
    gridlinesV = np.array([[gridv, -maxval, gridv, maxval] for gridv in gridvals])
    gridsegs = np.vstack((gridlinesH, gridlinesV))
    print 'np.shape(gridlines) = ', np.shape(gridsegs)
    # Make the bond linesegments
    xy = lat.xy
    bondsegs = np.array([[xy[b[0], 0], xy[b[0], 1], xy[b[1], 0], xy[b[1], 1]] for b in lat.BL])
    # get crossings
    print 'gridsegs = ', gridsegs
    does_intersect = lsegs.linesegs_intersect_linesegs(bondsegs, gridsegs)
    tmp_glat = GyroLattice(lat, lp)
    OmK = copy.deepcopy(tmp_glat.OmK)
    print 'Altering weak bonds --> ', args.weakbond_val
    for bond in lat.BL[does_intersect]:
        OmK[bond[0], np.where(lat.NL[bond[0]] == bond[1])] = args.weakbond_val
    glat = GyroLattice(lat, lp, OmK=OmK)
    glat.plot_OmK()
    if args.dice_eigval:
        glat.save_eigval_eigvect()
        glat.save_DOSmovie()


def localization(lp, args):
    """Load a periodic lattice from file, provide physics, and seek exponential localization of modes

    Example usage:
    python run_series.py -pro gyro_lattice_class -opts LT/hucentroid/-periodic/-NP/20/-localization/-save_eig -var AB 0.0:0.05:1.0
    python run_series.py -pro gyro_lattice_class -opts LT/hyperuniform/-periodic/-NP/50/-localization/-save_eig -var Vpin 0.1/0.5/1.0/2.0/4.0/6.0
    python run_series.py -pro gyro_lattice_class -opts LT/hyperuniform/-periodic/-NP/50/-DOSmovie/-save_eig -var Vpin 0.1/0.5/1.0/2.0/4.0/6.0

    """
    meshfn = le.find_meshfn(lp)
    lp['meshfn'] = meshfn
    lat = lattice_class.Lattice(lp)
    lat.load()
    extent = 2 * max(np.max(lat.xy[:, 0]), np.max(lat.xy[:, 1]))
    glat = GyroLattice(lat, lp)
    glat.save_localization(attribute=True, save_images=args.save_images, save_eigvect_eigval=args.save_eig)
    glat.plot_ill_dos(vmax=4. / extent, xlim=(0., 14.), ticks=[0, 2. / extent, 4. / extent],
                      cbar_ticklabels=[0, r'$2/L$', r'$4/L$'], cbar_labelpad=15)


def gap_scaling(lp, args):
    """

    Parameters
    ----------
    lp
    args

    Returns
    -------

    """
    par1 = 'delta'
    par1v = np.pi * np.arange(0.667, 1.25, 0.3)
    par2 = 'Omg'
    par2v = np.arange(1.0, 100., 10.) ** 2
    if args.N == 10:
        print 'set N == 10: choosing ev1, ev2'
        ev1 = 299
        ev2 = 300
    elif args.N == 8:
        print 'set N == 8: choosing ev1, ev2'
        ev1 = 191
        ev2 = 192
    elif args.N == 6:
        print 'set N == 6: choosing ev1, ev2'
        ev1 = 107
        ev2 = 108

    gapsz = np.zeros((len(par1v), len(par2v)), dtype=float)
    eigvals = np.zeros((2, len(par2v)), dtype=float)
    kk = 0
    for p1 in par1v:
        lp[par1] = p1
        lat = lattice_class.Lattice(lp)
        print 'Building lattice...'
        lat.build()

        diff = np.zeros(len(par2v))
        jj = 0
        for p2 in par2v:
            lp[par2] = p2
            glat = GyroLattice(lat, lp)
            eigval, eigvect = glat.eig_vals_vects(attribute=False)
            print 'eigval[ev1] = ', eigval[ev1]
            print 'eigval[ev2] = ', eigval[ev2]
            sys.exit()
            eigvals[0, jj] = np.imag(eigval[ev1])
            eigvals[1, jj] = np.imag(eigval[ev2])
            diff[jj] = np.abs(np.imag(eigval[ev1] - eigval[ev2]))
            jj += 1

        gapsz[kk] = diff
        plt.plot(par2v, np.log10(abs(diff)), '.-', label=str(par1v))
        plt.xlabel(r'$\Omega_g$')
        plt.ylabel(r'Gap width $\log_{10} W$')
        # plt.pause(0.01)
        plt.show()

        plt.clf()
        print 'par2v = ', par2v
        # print 'eigvals = ', eigvals
        plt.plot(par2v, eigvals[0], '.-')
        plt.plot(par2v, eigvals[1], '.-')
        plt.show()

        kk += 1

    plt.show()


def mode_scaling_tune_junction(lp, args, nmode=None):
    """First plot scaling for all modes as junction coupling is increased. Then, plot the nth mode as the
    scaling parameter evolves (typically the bond strength between particles that are nearer than some threshold).

    Example usage:
    python gyro_lattice_class.py -mode_scaling_tune_junction -LT hexjunction2triads -N 1 -OmKspec union0p000in0p100 -alph 0.1 -periodic
    python gyro_lattice_class.py -mode_scaling_tune_junction -LT spindle -N 2 -OmKspec union0p000in0p100 -alph 0.0 -periodic -aratio 0.1
    python gyro_lattice_class.py -mode_scaling_tune_junction -LT spindle -N 4 -OmKspec union0p000in0p100 -alph 0.0 -periodic -aratio 0.1

    python ./build/make_lattice.py -LT spindle -N 1 -periodic -check -skip_polygons -skip_gyroDOS -aratio 0.1
    python ./build/make_lattice.py -LT spindle -N 2 -periodic -check -skip_polygons -skip_gyroDOS -aratio 0.1
    python ./build/make_lattice.py -LT spindle -N 4 -periodic -skip_polygons -skip_gyroDOS -aratio 0.1
    python ./build/make_lattice.py -LT spindle -N 6 -periodic -skip_polygons -skip_gyroDOS -aratio 0.1 -check
    python ./build/make_lattice.py -LT spindle -N 8 -periodic -skip_polygons -skip_gyroDOS -aratio 0.1 -check

    Parameters
    ----------
    lp
    args
    nmode : int, int list, or None

    Returns
    -------

    """
    nkvals = 50
    if lp['LatticeTop'] in ['hexjunctiontriad', 'spindle', 'hexjunction2triads']:
        kvals = -np.unique(np.round(np.logspace(-1, 1., nkvals), 2))  # [::-1]
        dist_thres = lp['OmKspec'].split('union')[-1].split('in')[-1]
        lpmaster = copy.deepcopy(lp)
        lat = lattice_class.Lattice(lp)
        lat.load()
        if nmode is None:
            todo = np.arange(len(lat.xy[:, 0]))
        elif type(nmode) == int:
            todo = [nmode]
        else:
            todo = nmode

        ##########################################################################
        outfn = dio.prepdir(lat.lp['meshfn']) + 'glat_eigval_scaling_tune_junction'
        eigvals, first = [], True
        for (kval, dmyi) in zip(kvals, np.arange(len(kvals))):
            lp = copy.deepcopy(lpmaster)
            lp['OmKspec'] = 'union' + sf.float2pstr(kval, ndigits=3) + 'in' + dist_thres
            # lat = lattice_class.Lattice(lp)
            glat = GyroLattice(lat, lp)
            eigval, eigvect = glat.eig_vals_vects(attribute=True)
            if first:
                eigvals = np.zeros((len(kvals), len(eigval)), dtype=float)

            eigvals[dmyi, :] = np.imag(eigval)
            first = False

        # Plot flow of modes
        print 'eigvals = ', eigvals
        fig, axes = leplt.initialize_2panel_centy(Wfig=90, Hfig=65, x0frac=0.17, wsfrac=0.38)
        ax, ax1 = axes[0], axes[1]
        ymax = np.max(eigvals, axis=1)
        for kk in range(int(0.5 * len(eigvals[0])), len(eigvals[0])):
            ydat = eigvals[:, kk]
            ax.loglog(np.abs(kvals), ydat, 'b-')

        ax.set_ylim(ymin=0.1)
        ax.set_ylabel('frequency, $\omega$')
        ax.set_xlabel("coupling, $\Omega_k'$")
        if lp['LatticeTop'] == 'hexjunction2triads':
            ax.text(0.5, 0.9, 'Spectrum formation \n in double honeycomb junction',
                    ha='center', va='center', transform=fig.transFigure)
        elif lp['LatticeTop'] == 'spindle':
            if lp['NH'] == lp['NV']:
                nstr = str(int(lp['NH']))
            else:
                nstr = str(int(lp['NH'])) + ', ' + str(int(lp['NV']))
            ax.text(0.5, 0.9, 'Spectrum formation \n ' + r'in spindle lattice ($N=$' + nstr + ')',
                    ha='center', va='center', transform=fig.transFigure)

        lat.plot_BW_lat(fig=fig, ax=ax1, save=False, close=False, title='')
        plt.savefig(outfn + '_kmin' + sf.float2pstr(np.min(kvals)) + '_kmax' + sf.float2pstr(np.max(kvals)) + '.pdf')
        plt.show()
        ##########################################################################

        for ii in todo:
            modefn = dio.prepdir(lat.lp['meshfn']) + 'glat_mode_scaling_tune_junction_mode{0:05d}'.format(ii) +\
                     '_nkvals{0:04}'.format(nkvals) + '.mov'
            globmodefn = glob.glob(modefn)
            if not globmodefn:
                modedir = dio.prepdir(lat.lp['meshfn']) + 'glat_mode_scaling_tune_junction_mode{0:05d}/'.format(ii)
                dio.ensure_dir(modedir)
                previous_ev = None
                first = True
                dmyi = 0
                for kval in kvals:
                    lp = copy.deepcopy(lpmaster)
                    lp['OmKspec'] = 'union' + sf.float2pstr(kval, ndigits=3) + 'in' + dist_thres
                    # lat = lattice_class.Lattice(lp)
                    glat = GyroLattice(lat, lp)
                    eigval, eigvect = glat.eig_vals_vects(attribute=True)

                    # plot the nth mode
                    # fig, DOS_ax, eax = leplt.initialize_eigvect_DOS_header_plot(eigval, glat.lattice.xy,
                    #                                                             sim_type='gyro', cbar_nticks=2,
                    #                                                             cbar_tickfmt='%0.3f')
                    fig, dos_ax, eax, ax1, cbar_ax = \
                        leplt.initialize_eigvect_DOS_header_twinplot(eigval, glat.lattice.xy, sim_type='gyro',
                                                                     ax0_pos=[0.0, 0.10, 0.6, 0.60],
                                                                     ax1_pos=[0.6, 0.15, 0.3, 0.60],
                                                                     header_pos=[0.1, 0.78, 0.4, 0.20],
                                                                     xlabel_pad=8, fontsize=8)

                    # Get the theta that minimizes the difference between the present and previous eigenvalue
                    if previous_ev is not None:
                        realxy = np.real(previous_ev)
                        thetas, eigvects = gdh.phase_fix_nth_gyro(glat.eigvect, ngyro=0, basis='XY')
                        modenum = gdh.phase_difference_minimum(eigvects, realxy, basis='XY')
                        # print 'thetas = ', thetas
                        # if theta < 1e-9:
                        #     print 'problem with theta'
                        #     sys.exit()
                    else:
                        thetas, eigvects = gdh.phase_fix_nth_gyro(glat.eigvect, ngyro=0, basis='XY')
                        modenum = ii

                    glat.lattice.plot_BW_lat(fig=fig, ax=eax, save=False, close=False, axis_off=False, title='')
                    fig, [scat_fg, scat_fg2, pp, f_mark, lines_12_st], cw_ccw = \
                        leplt.construct_eigvect_DOS_plot(glat.lattice.xy, fig, dos_ax, eax, eigval, eigvects,
                                                         modenum, 'gyro', glat.lattice.NL, glat.lattice.KL,
                                                         marker_num=0, color_scheme='default', sub_lattice=-1,
                                                         amplify=1., title='')

                    # fig, ax_tmp, [scat_fg, pp, f_mark, lines12_st] = \
                    #     glat.plot_eigvect_excitation(ii, eigval=eigval, eigvect=eigvect, ax=eax, plot_lat=first,
                    #                                  theta=theta, normalization=1.0)  # color=lecmaps.blue())

                    # Store this current eigvector as 'previous_ev'
                    previous_ev = eigvects[modenum]

                    # scat_fg.remove()
                    # scat_fg2.remove()
                    # pp.remove()
                    # if f_mark is not None:
                    #     f_mark.remove()
                    # lines_12_st.remove()
                    # eax.cla()

                    # Plot where in evolution we are tracking
                    ngyros = int(np.shape(eigvals)[1] * 0.5)
                    halfev = eigvals[:, ngyros:]
                    for row in halfev.T:
                        ax1.loglog(np.abs(kvals), row, 'b-')

                    trackmark = ax1.plot(np.abs(kval), np.abs(np.imag(eigval))[modenum], 'ro')
                    ax1.set_xlabel(r"vertex coupling, $\Omega_k'$")
                    ax1.set_ylabel(r"frequency, $\omega$")

                    dos_ax.set_xlim(xmin=0)
                    plt.savefig(modedir + 'DOS_' + '{0:05}'.format(dmyi) + '.png', dpi=200)

                    dmyi += 1
                    first = False

                # Make movie
                imgname = modedir + 'DOS_'
                movname = modedir[:-1] + '_nkvals{0:04}'.format(nkvals)
                lemov.make_movie(imgname, movname, indexsz='05', framerate=5, rm_images=True, save_into_subdir=True,
                                 imgdir=modedir)


def polygon_phases_tune_junction(lp, args, nmode=None):
    """Plot the phase differences ccw around each polygon in the network for many glats as junction coupling
    is increased. Do this for the nth mode as the scaling parameter evolves
    (typically the bond strength between particles that are nearer than some threshold).

    Example usage:
    python gyro_lattice_class.py -polygon_phases_tune_junction -LT hexjunction2triads -N 1 -OmKspec union0p000in0p100 -alph 0.1 -periodic
    python gyro_lattice_class.py -polygon_phases_tune_junction -LT spindle -N 2 -OmKspec union0p000in0p100 -alph 0.0 -periodic -aratio 0.1
    python gyro_lattice_class.py -polygon_phases_tune_junction -LT spindle -N 4 -OmKspec union0p000in0p100 -alph 0.0 -periodic -aratio 0.1

    # for making lattices
    python ./build/make_lattice.py -LT spindle -N 4 -periodic -skip_gyroDOS -aratio 0.1

    Parameters
    ----------
    lp
    args
    nmode : int, int list, or None

    Returns
    -------

    """
    cmap = lecmaps.ensure_cmap('bbr0')
    nkvals = 50
    if lp['LatticeTop'] in ['hexjunctiontriad', 'spindle', 'hexjunction2triads']:
        kvals = -np.unique(np.round(np.logspace(-1, 1., nkvals), 2))  # [::-1]
        dist_thres = lp['OmKspec'].split('union')[-1].split('in')[-1]
        lpmaster = copy.deepcopy(lp)
        lat = lattice_class.Lattice(lp)
        lat.load()
        if nmode is None:
            todo = np.arange(len(lat.xy[:, 0]))
        elif type(nmode) == int:
            todo = [nmode]
        else:
            todo = nmode

        ##########################################################################
        # First collect eigenvalue flow
        eigvals, first = [], True
        for (kval, dmyi) in zip(kvals, np.arange(len(kvals))):
            lp = copy.deepcopy(lpmaster)
            lp['OmKspec'] = 'union' + sf.float2pstr(kval, ndigits=3) + 'in' + dist_thres
            # lat = lattice_class.Lattice(lp)
            glat = GyroLattice(lat, lp)
            eigval, eigvect = glat.eig_vals_vects(attribute=True)
            if first:
                eigvals = np.zeros((len(kvals), len(eigval)), dtype=float)

            eigvals[dmyi, :] = np.imag(eigval)
            first = False

        ##########################################################################

        lp = copy.deepcopy(lpmaster)
        glat = GyroLattice(lat, lp)
        # add meshfn without OmKspecunion part
        mfe = glat.lp['meshfn_exten']
        if mfe[0:13] == '_OmKspecunion':
            meshfnextenstr = mfe.split(mfe.split('_')[1])[-1]
        else:
            raise RuntimeError('Handle this case here -- should be easy: split meshfn_exten to pop OmKspec out')

        for ii in todo[::-1]:
            modefn = dio.prepdir(lat.lp['meshfn']) + 'glat_mode_phases_scaling_tune_junction_mode{0:05d}'.format(ii) +\
                     meshfnextenstr + '_nkvals{0:04}'.format(nkvals) + '.mov'
            globmodefn = glob.glob(modefn)
            if not globmodefn:
                modedir = dio.prepdir(lat.lp['meshfn']) + 'glat_mode_phases_tune_junction_mode{0:05d}'.format(ii) + \
                          meshfnextenstr + '/'
                dio.ensure_dir(modedir)
                previous_ev = None
                first = True
                dmyi = 0
                for kval in kvals:
                    lp = copy.deepcopy(lpmaster)
                    lp['OmKspec'] = 'union' + sf.float2pstr(kval, ndigits=3) + 'in' + dist_thres
                    # lat = lattice_class.Lattice(lp)
                    glat = GyroLattice(lat, lp)
                    eigval, eigvect = glat.eig_vals_vects(attribute=True)

                    # plot the nth mode
                    # fig, DOS_ax, eax = leplt.initialize_eigvect_DOS_header_plot(eigval, glat.lattice.xy,
                    #                                                             sim_type='gyro', cbar_nticks=2,
                    #                                                             cbar_tickfmt='%0.3f')
                    fig, dos_ax, eax, ax1, cbar_ax = \
                        leplt.initialize_eigvect_DOS_header_twinplot(eigval, glat.lattice.xy, sim_type='gyro',
                                                                     ax0_pos=[0.0, 0.10, 0.45, 0.55],
                                                                     ax1_pos=[0.65, 0.15, 0.3, 0.60],
                                                                     header_pos=[0.1, 0.78, 0.4, 0.20],
                                                                     xlabel_pad=8, fontsize=8)

                    cax = plt.axes([0.455, 0.10, 0.02, 0.55])

                    # Get the theta that minimizes the difference between the present and previous eigenvalue
                    # IN ORDER TO CONNECT MODES PROPERLY
                    if previous_ev is not None:
                        realxy = np.real(previous_ev)
                        thetas, eigvects = gdh.phase_fix_nth_gyro(glat.eigvect, ngyro=0, basis='XY')
                        # only look at neighboring modes
                        # (presumes sufficient resolution to disallow simultaneous crossings)
                        mmin = max(modenum - 2, 0)
                        mmax = min(modenum + 2, len(eigvects))
                        modenum = gdh.phase_difference_minimum(eigvects[mmin:mmax], realxy, basis='XY')
                        modenum += mmin
                        # print 'thetas = ', thetas
                        # if theta < 1e-9:
                        #     print 'problem with theta'
                        #     sys.exit()
                    else:
                        thetas, eigvects = gdh.phase_fix_nth_gyro(glat.eigvect, ngyro=0, basis='XY')
                        modenum = ii

                    # Plot the lattice with bonds
                    glat.lattice.plot_BW_lat(fig=fig, ax=eax, save=False, close=False, axis_off=False, title='')
                    # plot excitation
                    fig, [scat_fg, scat_fg2, pp, f_mark, lines_12_st], cw_ccw = \
                        leplt.construct_eigvect_DOS_plot(glat.lattice.xy, fig, dos_ax, eax, eigval, eigvects,
                                                         modenum, 'gyro', glat.lattice.NL, glat.lattice.KL,
                                                         marker_num=0, color_scheme='default', sub_lattice=-1,
                                                         amplify=1., title='')
                    # Plot the polygons colored by phase
                    polys = glat.lattice.get_polygons()
                    patches, colors = [], []
                    for poly in polys:
                        addv = np.array([0., 0.])
                        # build up positions, taking care of periodic boundaries
                        xys = np.zeros_like(glat.lattice.xy[poly], dtype=float)
                        xys[0] = glat.lattice.xy[poly[0]]
                        for (site, qq) in zip(poly[1:], range(len(poly) - 1)):
                            if latfns.bond_is_periodic(poly[qq], site, glat.lattice.BL):
                                toadd = latfns.get_periodic_vector(poly[qq], site,
                                                                   glat.lattice.PVx, glat.lattice.PVy,
                                                                   glat.lattice.NL, glat.lattice.KL)
                                if np.shape(toadd)[0] > 1:
                                    raise RuntimeError('Handle the case of multiple periodic bonds between ii jj here')
                                else:
                                    addv += toadd[0]
                            xys[qq + 1] = glat.lattice.xy[site] + addv
                            print 'site, poly[qq - 1] = ', (site, poly[qq])
                            print 'addv = ', addv

                        xys = np.array(xys)
                        polygon = Polygon(xys, True)
                        patches.append(polygon)

                        # Check the polygon
                        # plt.close('all')
                        # plt.plot(xys[:, 0], xys[:, 1], 'b-')
                        # plt.show()

                        # Get mean phase difference in this polygon
                        # Use weighted arithmetic mean of (cos(angle), sin(angle)), then take the arctangent.
                        yinds = 2 * np.array(poly) + 1
                        xinds = 2 * np.array(poly)
                        weights = glatfns.calc_magevecs_full(eigvect[modenum])
                        # To take mean, follow
                        # https://en.wikipedia.org/wiki/Mean_of_circular_quantities#Mean_of_angles
                        # with weights from
                        # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Mathematical_definition
                        # First take differences in angles
                        phis = np.arctan2(np.real(eigvects[modenum, yinds]), np.real(eigvects[modenum, xinds]))
                        print 'phis = ', phis
                        phis = np.mod(phis, np.pi * 2)
                        print 'phis = ', phis
                        # Now convert to vectors, take mean of both x and y components. Then grab atan2(y,x) of result.
                        xx, yy = np.mean(np.cos(np.diff(phis))), np.mean(np.sin(np.diff(phis)))
                        dphi = np.arctan2(yy, xx)
                        print 'dphi = ', dphi
                        colors.append(dphi)

                    # sys.exit()
                    pp = PatchCollection(patches, alpha=0.4, cmap=cmap)
                    pp.set_array(np.array(colors))
                    eax.add_collection(pp)
                    pp.set_clim([-np.pi, np.pi])
                    cbar = fig.colorbar(pp, cax=cax)

                    # Store this current eigvector as 'previous_ev'
                    previous_ev = eigvects[modenum]

                    # Plot where in evolution we are tracking
                    ngyros = int(np.shape(eigvals)[1] * 0.5)
                    halfev = eigvals[:, ngyros:]
                    for row in halfev.T:
                        ax1.loglog(np.abs(kvals), row, 'b-')

                    trackmark = ax1.plot(np.abs(kval), np.abs(np.imag(eigval))[modenum], 'ro')
                    ax1.set_xlabel(r"vertex coupling, $\Omega_k'$")
                    ax1.set_ylabel(r"frequency, $\omega$")
                    eax.xaxis.set_ticks([])
                    eax.yaxis.set_ticks([])
                    cbar.set_ticks([-np.pi, 0, np.pi])
                    cbar.set_ticklabels([r'-$\pi$', 0, r'$\pi$'])
                    cbar.set_label(r'phase, $\Delta \phi$')

                    dos_ax.set_xlim(xmin=0)
                    plt.savefig(modedir + 'DOS_' + '{0:05}'.format(dmyi) + '.png', dpi=200)

                    # remove plotted excitation
                    scat_fg.remove()
                    scat_fg2.remove()
                    pp.remove()
                    if f_mark is not None:
                        f_mark.remove()
                    lines_12_st.remove()
                    eax.cla()

                    dmyi += 1
                    first = False

                # Make movie
                imgname = modedir + 'DOS_'
                movname = modedir[:-1] + '_nkvals{0:04}'.format(nkvals)
                lemov.make_movie(imgname, movname, indexsz='05', framerate=5, rm_images=True, save_into_subdir=True,
                                 imgdir=modedir)


if __name__ == '__main__':
    # Fix a gyro movie
    dir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/networks/hucentroid/' + \
          'hucentroid_square_periodicstrip_d01_NH000050_NV000030_NP000050/gyro_thetatwistsweep_nthetas41/'
    ims = sorted(glob.glob(dir + 'manual_movie/*.png'))
    ii = 0
    for im in ims:
        if im.split('.png')[0][-5:] != '{0:05d}'.format(ii):
            newname = im[0:-9] + '{0:05d}'.format(ii) + '.png'
            print 'moving ', im, ' to ', newname
            subprocess.call(['mv', im, newname])
        ii += 1

    imgname = im[0:-9]
    movname = dir + 'thetatwist_spiral.mov'
    lemov.make_movie(imgname, movname, framerate=9)