import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import lepm.plotting.plotting as leplt
import lepm.plotting.movies as lemov
import lepm.dataio as dio
import numpy.linalg as la
import lepm.plotting.colormaps as lecmaps

'''Visualize haldane model
python ./script_haldane_model.py

Knobs to tune:
turn on dual_panel to make figures that show how the gap size scales with t2, DeltaAB.
turn on plot3d to plot the dispersion in 3d
turn on nocolor to plot dispersion with bands colored as (band1color, band2color) rather than using berry curvature
'''

band1color = '#90354c'
band2color = '#0A6890'
graycolor = '#000000'


def h0_func(coska, sinka):
    """h0 = t1 sum[sigma_x cos(k ai) - sigma_y sin(k ai)]"""
    return sx * coska - sy * sinka


def ht2_func(coska, sinka, sinkb, t1, t2):
    """Haldane model with NN hoppings and imaginary NNN hoppings"""
    return t1 * h0_func(coska, sinka) + 2*t2 * sz * sinkb


def energy_berrycurv_ht2(coska, sinka, sinkb, nrows, ncols, t1, t2, mm):
    """Return the eigenvalues and berry curvature at given value of ka

    Parameters
    ----------
    coska :
    sinka :
    sinkb :
    nrows : int
    ncols : int
    t1 : float
        NN real hopping
    t2 : float
        NNN complex hopping
    mm : float
        onsite energy splitting

    Returns
    -------
    """
    # Would be, if not vectorized:
    # energy = np.abs(linalg.det(ht2_func(coska, sinka, sinkb, t1, t2)))
    # Form a len(kv) x 4 hamiltonian, where columns are (0,0) (0,1) (1,0) (1,1)

    # Pauli matrices
    s0 = np.identity(2)
    sx = np.array([[0, 1], [1, 0]])
    sy = np.array([[0, -1j], [1j, 0]])
    sz = np.diag([1, -1])

    # ham is the haldane matrix, evaluated at this kx ky site, as a 1 x 4 vector
    if isinstance(coska, float):
        ham = np.zeros((1, 4), dtype=complex)
    else:
        ham = np.zeros((len(coska), 4), dtype=complex)
    ham[:, 1] += t1 * (sx[0, 1] * coska - sy[0, 1] * sinka)
    ham[:, 2] += t1 * (sx[1, 0] * coska - sy[1, 0] * sinka)
    ham[:, 0] += 2. * t2 * sz[0, 0] * sinkb + mm
    ham[:, 3] += 2. * t2 * sz[1, 1] * sinkb - mm

    det = ham[:, 0] * ham[:, 3] - ham[:, 1] * ham[:, 2]
    tr = ham[:, 0] + ham[:, 3]
    energy1 = tr * 0.5 + np.sqrt(tr ** 2 - 4. * det) * 0.5
    energy2 = tr * 0.5 - np.sqrt(tr ** 2 - 4. * det) * 0.5

    # Now compute Berry curvature at each site
    # method for computing the Berry curvature in k space that is gauge invariant:
    # http://iopscience.iop.org/article/10.1088/1367-2630/16/7/073016/pdf
    berry1 = np.zeros_like(coska)
    berry2 = np.zeros_like(coska)
    for ii in range(nrows):
        # # this is a hack since two rows are problematic
        # if (ii in [23, 87] and nrows == 113) or (ii in [20, 77] and nrows == 100):
        #     for jj in range(ncols):
        #         ij = ii * ncols + jj
        #         berry1[ij] = berry1[(ii - 1) * ncols + jj]
        #         berry2[ij] = berry2[(ii - 1) * ncols + jj]
        # else:
        for jj in range(ncols):
            # form the ij hamiltonian and diagonalize it
            # translate ii, jj to ij, which is the raveled index
            ij = ii * ncols + jj
            xj = ii * ncols + ((jj + 1) % ncols)
            iy = ((ii + 1) % nrows) * ncols + jj
            xy = ((ii + 1) % nrows) * ncols + ((jj + 1) % ncols)
            hij = np.array([[ham[ij, 0], ham[ij, 1]], [ham[ij, 2], ham[ij, 3]]])
            hxj = np.array([[ham[xj, 0], ham[xj, 1]], [ham[xj, 2], ham[xj, 3]]])
            hiy = np.array([[ham[iy, 0], ham[iy, 1]], [ham[iy, 2], ham[iy, 3]]])
            hxy = np.array([[ham[xy, 0], ham[xy, 1]], [ham[xy, 2], ham[xy, 3]]])
            w, uij = la.eig(hij)
            w, uxj = la.eig(hxj)
            w, uiy = la.eig(hiy)
            w, uxy = la.eig(hxy)
            # uij0, uij1 = uij[:, 0], uij[:, 1]
            # uxj0, uxj1 = uxj[:, 0], uxj[:, 1]
            # uiy0, uiy1 = uiy[:, 0], uiy[:, 1]
            # uxy0, uxy1 = uxy[:, 0], uxy[:, 1]
            # print 'uij = ', uij
            # print 'w = ', w
            uij0, uij1 = uij[0], uij[1]
            uxj0, uxj1 = uxj[0], uxj[1]
            uiy0, uiy1 = uiy[0], uiy[1]
            uxy0, uxy1 = uxy[0], uxy[1]

            # Uxkij is U_x(k) = <u_ij | u(k+dkx)>
            Uxkij = np.array([np.dot(np.conj(uij0), uxj0), np.dot(np.conj(uij1), uxj1)])
            # Uykij is U_y(k) = <u_ij | u(k+dky) >
            Uykij = np.array([np.dot(np.conj(uij0), uiy0), np.dot(np.conj(uij1), uiy1)])
            # Uykxj is U_y(k + delta_{k_x}) = <u(k + dkx) | u(k + dkx + dky)>
            Uykxj = np.array([np.dot(np.conj(uxj0), uxy0), np.dot(np.conj(uxj1), uxy1)])
            # Uxkiy is U_x(k + delta_{k_y}) =  <u(k + dky) | u(k + dkx + dky)>
            Uxkiy = np.array([np.dot(np.conj(uiy0), uxy0), np.dot(np.conj(uiy1), uxy1)])

            berry1[ij] = (np.imag(np.log(Uxkij[0] * Uykxj[0] / (Uxkiy[0] * Uykij[0]))) + np.pi) % (2. * np.pi) - np.pi
            berry2[ij] = (np.imag(np.log(Uxkij[1] * Uykxj[1] / (Uxkiy[1] * Uykij[1]))) + np.pi) % (2. * np.pi) - np.pi

            if berry1[ij] > 0.1:
                print 'ij = ', ij
                print 'ii = ', ii
                print 'jj = ', jj
                print 'nrows, ncols = ', nrows, ncols
            # if berry2[ij] > np.pi:
            #     berry2[ij] -= 2. * np.pi

    return energy1, energy2, berry1, berry2


def energy_ht2(coska, sinka, sinkb, t1, t2, mm):
    """Return the eigenvalues """
    # Would be, if not vectorized:
    # energy = np.abs(linalg.det(ht2_func(coska, sinka, sinkb, t1, t2)))
    # Form a len(kv) x 4 hamiltonian, where columns are (0,0) (0,1) (1,0) (1,1)

    # Pauli matrices
    s0 = np.identity(2)
    sx = np.array([[0, 1], [1, 0]])
    sy = np.array([[0, -1j], [1j, 0]])
    sz = np.diag([1, -1])

    # ham is the haldane matrix, evaluated at this kx ky site, as a 1 x 4 vector
    if isinstance(coska, float):
        ham = np.zeros((1, 4), dtype=complex)
    else:
        ham = np.zeros((len(coska), 4), dtype=complex)
    ham[:, 1] += t1 * (sx[0, 1] * coska - sy[0, 1] * sinka)
    ham[:, 2] += t1 * (sx[1, 0] * coska - sy[1, 0] * sinka)
    ham[:, 0] += 2. * t2 * sz[0, 0] * sinkb + mm
    ham[:, 3] += 2. * t2 * sz[1, 1] * sinkb - mm
    det = ham[:, 0] * ham[:, 3] - ham[:, 1] * ham[:, 2]
    tr = ham[:, 0] + ham[:, 3]
    energy1 = tr*0.5 + np.sqrt(tr**2 - 4.*det)*0.5
    energy2 = tr*0.5 - np.sqrt(tr**2 - 4.*det)*0.5
    return energy1, energy2


def haldane_dispersion(nx, ny, t1, t2, mm):
    """"""
    # Vectorize lattice vectors a, and momentum vecs kv
    avecs = np.array([[1, 0],
                      [np.cos(np.pi*2./3.), np.sin(np.pi*2./3.)],
                      [np.cos(-np.pi*2./3.), np.sin(-np.pi*2./3.)]])
    bvecs = np.array([[0, 2. * np.sin(np.pi/3.)],
                      [-1. - np.cos(np.pi/3.), -np.sin(np.pi/3.)],
                      [1. + np.cos(-np.pi/3.), -np.sin(-np.pi/3.)]])
    km = np.meshgrid(np.linspace(-np.pi, np.pi + 2.*np.pi/float(nx), nx),
                     np.linspace(-np.pi, np.pi + 2.*np.pi/float(ny), ny))
    kv = np.dstack((km[0].ravel(), km[1].ravel()))[0]

    coska = np.sum(np.cos(np.dot(kv, avecs.T)), axis=1)
    sinka = np.sum(np.sin(np.dot(kv, avecs.T)), axis=1)
    sinkb = np.sum(np.sin(np.dot(kv, bvecs.T)), axis=1)

    energy1, energy2 = energy_ht2(coska, sinka, sinkb, t1, t2, mm)
    return km, kv, np.real(energy1), np.real(energy2)


def haldane_dispersion_berry(nn, t1, t2, mm):
    """Compute the energy versus kx, ky but also compute the berry curvature at each k point
    """
    # Vectorize lattice vectors a, and momentum vecs kv
    avecs = np.array([[1, 0],
                      [np.cos(np.pi*2./3.), np.sin(np.pi*2./3.)],
                      [np.cos(-np.pi*2./3.), np.sin(-np.pi*2./3.)]])
    bvecs = np.array([[0, 2. * np.sin(np.pi/3.)],
                      [-1. - np.cos(np.pi/3.), -np.sin(np.pi/3.)],
                      [1. + np.cos(-np.pi/3.), -np.sin(-np.pi/3.)]])
    karr = np.linspace(-np.pi, np.pi + 2. * np.pi / float(nn), nn)
    km = np.meshgrid(karr, karr)
    kv = np.dstack((km[0].ravel(), km[1].ravel()))[0]

    coska = np.sum(np.cos(np.dot(kv, avecs.T)), axis=1)
    sinka = np.sum(np.sin(np.dot(kv, avecs.T)), axis=1)
    sinkb = np.sum(np.sin(np.dot(kv, bvecs.T)), axis=1)

    # print 'np.shape(sinka) = ', np.shape(sinka)
    # print 'np.shape(km) = ', np.shape(km)
    nrows, ncols = np.shape(km[0])
    energy1, energy2, berry1, berry2 = energy_berrycurv_ht2(coska, sinka, sinkb, nrows, ncols, t1, t2, mm)

    # plt.clf()
    # cberry1 = (berry1 + np.pi) / (2. * np.pi)
    # plt.scatter(kv[:, 0], kv[:, 1], c=cberry1, vmin=0.49, vmax=0.51, edgecolor='none',
    #             cmap=lecmaps.ensure_cmap('bbr0'))
    # plt.show()
    # sys.exit()
    return km, kv, np.real(energy1), np.real(energy2), berry1, berry2


def movie_varyt2(t1, t2arr, mm, nx, ny, maindir, outdir, fontsize=20, tick_fontsize=16, saveims=True,
                 dual_panel=False, color_berry=False, cmap='bbr0', vmin=0.499, vmax=0.501, plot3d=False):
    """Plot the band structure of the haldane model as we vary the NNN hopping

    Parameters
    ----------
    t1 : float
        magnitude of real NN hopping
    t2 : float
        magnitude of complex NNN hopping
    dab_arr : n x 1 float array
        the DeltaAB values to use for each frame
    nn : int
        number of kx values (same as # ky values)
    maindir : str
        where to save the movie
    outdir : str
        where to save the images
    saveims : bool
        overwrite the existing images
    dual_panel : bool
        plot the 2d dispersion with calibration of gap opening as function of DeltaAB
    vmin : float
        the minimimum color value from 0 to 1, if color_berry=True
    vmax : float
        the maximum color value from 0 to 1, if color_berry = True
    plot3d : bool
        whether to plot the dispersion in 3d or not
    """
    kk = 0
    gaps = []
    t2s = []
    if isinstance(cmap, str):
        cmap = lecmaps.ensure_cmap(cmap)

    for t2 in t2arr:
        if color_berry:
            km, kv, energy1, energy2, berry1, berry2 = haldane_dispersion_berry(nx, t1, t2, mm)
        else:
            km, kv, energy1, energy2 = haldane_dispersion(nx, ny, t1, t2, mm)
        gap = np.min(energy1) * 2.
        print 'gap = ', gap
        gaps.append(gap)
        t2s.append(t2)

        if saveims:
            if dual_panel:
                fig, ax = leplt.initialize_2panel_3o4ar_cent(fontsize=fontsize, x0frac=0.085)
                ax, ax2 = ax[0], ax[1]
            else:
                fig, ax = leplt.initialize_1panel_centered_fig(Wfig=180, Hfig=180, wsfrac=0.7, fontsize=fontsize)

            if not color_berry:
                if t2 > mm / np.sqrt(3.):
                    ax.plot(km[1], energy1.reshape(np.shape(km[0])), '-', color=band1color)
                    ax.plot(km[1], energy2.reshape(np.shape(km[0])), '-', color=band2color)
                elif t2 < - mm / np.sqrt(3.):
                    ax.plot(km[1], energy1.reshape(np.shape(km[0])), '-', color=band2color)
                    ax.plot(km[1], energy2.reshape(np.shape(km[0])), '-', color=band1color)
                else:
                    ax.plot(km[1], energy1.reshape(np.shape(km[0])), '-', color=graycolor)
                    ax.plot(km[1], energy2.reshape(np.shape(km[0])), '-', color=graycolor)
            else:
                b1color = berry1.reshape(np.shape(km[0])) / (2. * np.pi) + 0.5
                b2color = berry2.reshape(np.shape(km[0])) / (2. * np.pi) + 0.5
                print 'b1color=', b1color
                # sys.exit()
                energy1 = energy1.reshape(np.shape(km[0]))
                energy2 = energy2.reshape(np.shape(km[0]))
                # ax.scatter([0., 0.5, 1.0], [0., 0.5, 1.0], c=[0., 0.5, 1.0], edgecolor='none', cmap=cmap, alpha=1)
                for ii in range(len(km[0])):
                    ax.scatter(km[1][ii], energy1[ii], c=b1color, edgecolor='none', cmap=cmap, alpha=1.,
                               vmin=vmin, vmax=vmax)
                    ax.scatter(km[1][ii], energy2[ii], c=b2color, edgecolor='none', cmap=cmap, alpha=1.,
                               vmin=vmin, vmax=vmax)
                # plt.show()
                # sys.exit()

            ax.set_xlabel(r'$k_x$', labelpad=15, fontsize=fontsize)
            ax.set_ylabel(r'$\omega$', labelpad=15, fontsize=fontsize, rotation=0)
            ax.xaxis.set_ticks([-np.pi, 0, np.pi])
            ax.xaxis.set_ticklabels([r'-$\pi$', 0, r'$\pi$'], fontsize=tick_fontsize)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(tick_fontsize)
            ax.axis('scaled')
            ax.set_xlim(-np.pi, np.pi)
            ax.set_ylim(-np.pi, np.pi)
            # title = r'$\Delta \omega =$' + '{0:0.3f}'.format(gap) + r' $t_1$'
            title = r'$t_2 =$' + '{0:0.3f}'.format(t2) + r' $t_1$,   ' + \
                    r'$\Delta_{AB} =$' + '{0:0.3f}'.format(mm) + r' $t_1$'
            ax.text(0.5, 1.1, title, ha='center', va='center', fontsize=fontsize, transform=ax.transAxes)

            if dual_panel:
                # Make second figure
                ax2.plot()
                ax2.set_xlim(0, 0.2)
                ax2.set_ylim(0, 0.7)
                ax2.plot([np.min(t2arr), np.max(t2arr)], [0.0, 3.36864398885*np.max(t2arr)], '-', color='#a1bfdb', lw=3)
                ax2.scatter(t2s, gaps, color='k', zorder=9999999999)
                title = 'Gap size: ' + r'$\Delta\omega/t_1 \approx 3.369\, t_2$'
                ax2.text(0.5, 1.1, title, ha='center', va='center', fontsize=fontsize, transform=ax2.transAxes)
                for tick in ax2.xaxis.get_major_ticks():
                    tick.label.set_fontsize(tick_fontsize)
                for tick in ax2.yaxis.get_major_ticks():
                    tick.label.set_fontsize(tick_fontsize)
                ax2.set_xlabel(r'$t_2 = \Omega_k^2/8\Omega_g$', fontsize=fontsize, labelpad=15)
                ax2.set_ylabel(r'$\Delta\omega/t_1 = 2 \Delta\omega/\Omega_k$', fontsize=fontsize, rotation=90, labelpad=35)

            # Save the image
            plt.savefig(outdir + 'dispersion_{0:04d}'.format(kk) + '.png', dpi=140)
            plt.close('all')
            kk += 1
            print 'dirac = ', 2.*np.pi/3., ', ', -2.*np.pi/(3.*np.sqrt(3.))
            ind = np.argmin(energy1)
            dirac = kv[ind]
            print 'dirac where = ', dirac

    print 't2s = ', t2s
    print 'gaps = ', gaps
    zz = np.polyfit(np.array(t2s), np.array(gaps), 1)
    pp = np.poly1d(zz)
    print pp
    print zz[0]
    print zz[1]

    imgname = outdir + 'dispersion_'
    movname = maindir + 'dispersion_varyt2'
    movname += '_nkx{0:06d}'.format(nx) + '_nky{0:06d}'.format(ny)
    lemov.make_movie(imgname, movname, indexsz='04', framerate=15)


def movie_varyDeltaAB(t1, t2, dab_arr, nx, ny, maindir, outdir, fontsize=20, tick_fontsize=16,
                      saveims=True, dual_panel=False, color_berry=False, cmap='coolwarm',
                      vmin=0.49, vmax=0.51, plot3d=False):
    """Create a movie of band structure with varying DeltaAB

    Parameters
    ----------
    t1 : float
        magnitude of real NN hopping
    t2 : float
        magnitude of complex NNN hopping
    dab_arr : n x 1 float array
        the DeltaAB values to use for each frame
    nn : int
        number of kx values (same as # ky values)
    maindir : str
        where to save the movie
    outdir : str
        where to save the images
    saveims : bool
        overwrite the existing images
    dual_panel : bool
        plot the 2d dispersion with calibration of gap opening as function of DeltaAB
    vmin : float
        the minimimum color value from 0 to 1, if color_berry=True
    vmax : float
        the maximum color value from 0 to 1, if color_berry = True
    plot3d : bool
        whether to plot the dispersion in 3d or not
    """
    if isinstance(cmap, str):
        cmap = lecmaps.ensure_cmap(cmap)

    kk = 0
    gaps = []
    dabs = []
    for mm in dab_arr:
        if color_berry:
            km, kv, energy1, energy2, berry1, berry2 = haldane_dispersion_berry(nx, ny, t1, t2, mm)
        else:
            km, kv, energy1, energy2 = haldane_dispersion(nx, ny, t1, t2, mm)

        gap = np.min(energy1) * 2.
        print 'gap = ', gap
        gaps.append(gap)
        dabs.append(mm)

        if saveims:
            if not plot3d:
                if dual_panel:
                    fig, ax = leplt.initialize_2panel_3o4ar_cent(fontsize=fontsize, x0frac=0.085)
                    ax, ax2 = ax[0], ax[1]

                    # Make second figure
                    ax2.plot()
                    ax2.set_xlim(0, 0.2)
                    ax2.set_ylim(0, 0.7)
                    ax2.plot([np.min(t2arr), np.max(t2arr)], [0.0, 2*np.max(t2arr)], '-', color='#a1bfdb', lw=3)
                    ax2.scatter(dabs, gaps, color='k', zorder=9999999999)
                    title = 'Gap size: ' + r'$\Delta\omega/t_1 \approx 2 \, \Delta_{AB}$'
                    ax2.text(0.5, 1.1, title, ha='center', va='center', fontsize=fontsize, transform=ax2.transAxes)
                    for tick in ax2.xaxis.get_major_ticks():
                        tick.label.set_fontsize(tick_fontsize)
                    for tick in ax2.yaxis.get_major_ticks():
                        tick.label.set_fontsize(tick_fontsize)
                    ax2.set_xlabel(r'$t_2 = \Omega_k^2/8\Omega_g$', fontsize=fontsize, labelpad=15)
                    ax2.set_ylabel(r'$\Delta\omega/t_1 = 2 \Delta\omega/\Omega_k$',
                                   fontsize=fontsize, rotation=90, labelpad=35)
                else:
                    fig, ax = leplt.initialize_1panel_centered_fig(Wfig=180, Hfig=180, wsfrac=0.7, fontsize=fontsize)

                if not color_berry:
                    # check if topological or not
                    print 't2 = ', t2
                    print 'mm = ', mm * (3. * np.sqrt(3.))
                    if t2 > mm / np.sqrt(3.):
                        # yes, topological
                        ax.plot(km[1], energy1.reshape(np.shape(km[0])), '-', color=band1color)
                        ax.plot(km[1], energy2.reshape(np.shape(km[0])), '-', color=band2color)
                    elif t2 < - mm / np.sqrt(3.):
                        # flipped topology
                        ax.plot(km[1], energy1.reshape(np.shape(km[0])), '-', color=band2color)
                        ax.plot(km[1], energy2.reshape(np.shape(km[0])), '-', color=band1color)
                    else:
                        # not topological
                        ax.plot(km[1], energy1.reshape(np.shape(km[0])), '-', color=graycolor)
                        ax.plot(km[1], energy2.reshape(np.shape(km[0])), '-', color=graycolor)
                else:
                    b1color = berry1.reshape(np.shape(km[0])) / (2. * np.pi) + 0.5
                    b2color = berry2.reshape(np.shape(km[0])) / (2. * np.pi) + 0.5

                    energy1 = energy1.reshape(np.shape(km[0]))
                    energy2 = energy2.reshape(np.shape(km[0]))
                    # ax.scatter([0., 0.5, 1.0], [0., 0.5, 1.0], c=[0., 0.5, 1.0], edgecolor='none', cmap=cmap, alpha=1)
                    for ii in range(len(km[0])):
                        ax.scatter(km[1][ii], energy1[ii], c=b1color[ii], edgecolor='none', cmap=cmap, alpha=0.1,
                                   vmin=vmin, vmax=vmax)
                        ax.scatter(km[1][ii], energy2[ii], c=b2color[ii], edgecolor='none', cmap=cmap, alpha=0.1,
                                   vmin=vmin, vmax=vmax)
                    plt.show()
                    sys.exit()

                ax.set_ylabel(r'$\omega$', labelpad=15, fontsize=fontsize, rotation=0)
                # title = r'$\Delta \omega =$' + '{0:0.3f}'.format(gap) + r' $t_1$'
                title = r'$t_2 =$' + '{0:0.3f}'.format(t2) + r' $t_1$,   ' + \
                        r'$\Delta_{AB} =$' + '{0:0.3f}'.format(mm) + r' $t_1$'
                ax.text(0.5, 1.1, title, ha='center', va='center', fontsize=fontsize, transform=ax.transAxes)

            else:
                # Plot it in 3d
                fig, ax = leplt.initialize_1panel_centered_fig(Wfig=180, Hfig=180, wsfrac=0.7, fontsize=fontsize)
                ax = fig.gca(projection='3d')
                # Reshape colors, energies
                b1color = cmap((berry1.reshape(np.shape(km[0])) / (2. * np.pi)) / (vmax - vmin) + 0.5)
                b2color = cmap((berry2.reshape(np.shape(km[0])) / (2. * np.pi)) / (vmax - vmin) + 0.5)
                energy1 = energy1.reshape(np.shape(km[0]))
                energy2 = energy2.reshape(np.shape(km[0]))
                print 'b1color = ', b1color

                # Plot surfaces
                surf = ax.plot_surface(km[0], km[1], energy1, facecolors=b1color, rstride=1, cstride=1,
                                       vmin=vmin, vmax=vmax, cmap=cmap,
                                       linewidth=1, antialiased=False)
                surf = ax.plot_surface(km[0], km[1], energy2, facecolors=b2color, rstride=1, cstride=1,
                                       vmin=vmin, vmax=vmax, cmap=cmap,
                                       linewidth=1, antialiased=False)
                ax.set_ylabel(r'$k_y$', labelpad=15, fontsize=fontsize)
                ax.yaxis.set_ticks([-np.pi, 0, np.pi])
                ax.yaxis.set_ticklabels([r'-$\pi$', 0, r'$\pi$'], fontsize=tick_fontsize)
                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(tick_fontsize)
                ax.set_zlabel(r'$\omega$', labelpad=15, fontsize=fontsize, rotation=90)
                # ax.zaxis.set_ticks([-np.pi, 0, np.pi])
                # ax.zaxis.set_ticklabels([r'-$\pi$', 0, r'$\pi$'], fontsize=tick_fontsize)
                for tick in ax.zaxis.get_major_ticks():
                    tick.label.set_fontsize(tick_fontsize)
                ax.axis('scaled')
                ax.set_zlim(-np.pi, np.pi)

            ax.set_xlabel(r'$k_x$', labelpad=15, fontsize=fontsize)
            ax.xaxis.set_ticks([-np.pi, 0, np.pi])
            ax.xaxis.set_ticklabels([r'-$\pi$', 0, r'$\pi$'], fontsize=tick_fontsize)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(tick_fontsize)
            ax.axis('scaled')
            ax.set_xlim(-np.pi, np.pi)
            ax.set_ylim(-np.pi, np.pi)

            # Save it
            plt.savefig(outdir + 'dispersion_{0:04d}'.format(kk) + '.png'.format(t2), dpi=140)
            plt.close('all')
            kk += 1
            print 'dirac = ', 2. * np.pi / 3., ', ', -2. * np.pi / (3. * np.sqrt(3.))
            ind = np.argmin(energy1)
            dirac = kv[ind]
            print 'dirac where = ', dirac

        # print the fitting
        print 'dabs = ', dabs
        print 'gaps = ', gaps
        zz = np.polyfit(np.array(dabs), np.array(gaps), 1)
        pp = np.poly1d(zz)
        print 'pp = ', pp
        print 'zz[0] = ', zz[0]
        print 'zz[1] = ', zz[1]

    imgname = outdir + 'dispersion_'
    movname = maindir + 'dispersion_varyDeltaAB_{0:0.3f}'.format(t2).replace('.', 'p')
    movname += '_nkx{0:06d}'.format(nx) + '_nky{0:06d}'.format(ny)
    lemov.make_movie(imgname, movname, indexsz='04', framerate=15)


# Parameters
# number of sampling points
nx = 200
ny = 300
full3dplot = False
saveims = True
fontsize = 20
tick_fontsize = 16
maindir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/haldane_model/'
outdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/haldane_model/t2_movie/'
outdir_dab = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/haldane_model/DeltaAB_movie/'
outdir_dabt2 = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/haldane_model/DeltaAB_movie_t20p20/'
dio.ensure_dir(outdir)
dio.ensure_dir(outdir_dab)
dio.ensure_dir(outdir_dabt2)
# hopping strengths
t1 = 1.
t2max = 0.20
t2arr = np.arange(0, t2max + 1e-9, 0.01)
mmarr = np.arange(0, 3 * t2max + 1e-9, 0.01)
mm = 0.

# Create movie and image sequence
movie_varyDeltaAB(t1, 0., mmarr, nx, ny, maindir, outdir_dab,
                  fontsize=fontsize, tick_fontsize=tick_fontsize, saveims=saveims)
# movie_varyt2(t1, t2arr, mm, nx, ny, maindir, outdir, fontsize=fontsize, tick_fontsize=tick_fontsize, saveims=saveims)
# movie_varyDeltaAB(t1, t2max, mmarr, nx, ny, maindir, outdir_dabt2,
#                   fontsize=fontsize, tick_fontsize=tick_fontsize, saveims=saveims)

# Consider effective mass
# meff = t2/2 sum_{r=1 to 6} ((-1)^r cos(phi) -/+ sqrt(3) sin(phi))
# where minus sign is for one Dirac point and plus is for the other
# For phi_{r+3} = phi_r, this simplifies to
# meff = -/+ sqrt(3)/2 t2 sum_r phi_r
phis = 2. * np.pi * np.array([4./3., 4./3., 4./3., 4./3., 4./3., 4./3.])
phis = np.mod(phis, 2*np.pi)
msum = np.sum(np.sin(phis))
# msum = -np.cos(phis[0]) - np.sqrt(3) * np.sin(phis[0])
# msum += np.cos(phis[1]) - np.sqrt(3) * np.sin(phis[1])
# msum += -np.cos(phis[2]) - np.sqrt(3) * np.sin(phis[2])
# msum += np.cos(phis[3]) - np.sqrt(3) * np.sin(phis[3])
# msum += -np.cos(phis[4]) - np.sqrt(3) * np.sin(phis[4])
# msum += np.cos(phis[5]) - np.sqrt(3) * np.sin(phis[5])
print 'msum = ', msum
meff = t2arr * 0.5 * np.sqrt(3) * msum
gap = 2. * meff
print gap

if full3dplot:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(km[0], km[1], energy1.reshape(np.shape(km[0])), rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    surf = ax.plot_surface(km[0], km[1], energy2.reshape(np.shape(km[0])), rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    plt.xlabel(r'$k_x$', labelpad=20)
    plt.ylabel(r'$k_y$', labelpad=20)
    # plt.zlabel(r'$\omega$', labelpad=10)
    plt.show()
