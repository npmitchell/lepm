import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import lepm.plotting.plotting as leplt
import lepm.plotting.movies as lemov

'''Visualize haldane model'''


def h0_func(coska, sinka):
    """h0 = t1 sum[sigma_x cos(k ai) - sigma_y sin(k ai)]"""
    return sx * coska - sy * sinka


def ht2_func(coska, sinka, sinkb, t1, t2):
    """Haldane model with NN hoppings and imaginary NNN hoppings"""
    return t1 * h0_func(coska, sinka) + 2*t2 * sz * sinkb


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


def haldane_dispersion(nn, t1, t2, mm):
    """"""
    # Vectorize lattice vectors a, and momentum vecs kv
    avecs = np.array([[1, 0],
                      [np.cos(np.pi*2./3.), np.sin(np.pi*2./3.)],
                      [np.cos(-np.pi*2./3.), np.sin(-np.pi*2./3.)]])
    bvecs = np.array([[0, 2. * np.sin(np.pi/3.)],
                      [-1. - np.cos(np.pi/3.), -np.sin(np.pi/3.)],
                      [1. + np.cos(-np.pi/3.), -np.sin(-np.pi/3.)]])
    km = np.meshgrid(np.linspace(-np.pi, np.pi + 2.*np.pi/float(nn), nn),
                     np.linspace(-np.pi, np.pi + 2.*np.pi/float(nn), nn))
    kv = np.dstack((km[0].ravel(), km[1].ravel()))[0]

    coska = np.sum(np.cos(np.dot(kv, avecs.T)), axis=1)
    sinka = np.sum(np.sin(np.dot(kv, avecs.T)), axis=1)
    sinkb = np.sum(np.sin(np.dot(kv, bvecs.T)), axis=1)

    energy1, energy2 = energy_ht2(coska, sinka, sinkb, t1, t2, mm)
    return km, kv, np.real(energy1), np.real(energy2)


def movie_varyt2(t1, t2arr, mm, nn, maindir, outdir, fontsize=20, tick_fontsize=16, saveims=True):
    kk = 0
    gaps = []
    t2s = []
    for t2 in t2arr:
        km, kv, energy1, energy2 = haldane_dispersion(nn, t1, t2, mm)
        gap = np.min(energy1) * 2.
        print 'gap = ', gap
        gaps.append(gap)
        t2s.append(t2)

        if saveims:
            fig, ax = leplt.initialize_2panel_3o4ar_cent(fontsize=fontsize, x0frac=0.085)
            ax, ax2 = ax[0], ax[1]
            ax.plot(km[1], energy1.reshape(np.shape(km[0])), '-', color='#90354c')
            ax.plot(km[1], energy2.reshape(np.shape(km[0])), '-', color='#0A6890')
            ax.set_xlabel(r'$k_x$', labelpad=15, fontsize=fontsize)
            ax.set_ylabel(r'$\omega$', labelpad=15, fontsize=fontsize, rotation=0)
            ax.xaxis.set_ticks([-np.pi, 0, np.pi])
            ax.xaxis.set_ticklabels([r'-$\pi$', 0, r'$\pi$'], fontsize=tick_fontsize)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(tick_fontsize)
            ax.axis('scaled')
            ax.set_xlim(-np.pi, np.pi)
            ax.set_ylim(-np.pi, np.pi)
            title = r'$\Delta \omega =$' + '{0:0.3f}'.format(gap) + r' $t_1$'
            ax.text(0.5, 1.1, title, ha='center', va='center', fontsize=fontsize, transform=ax.transAxes)

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
            plt.savefig(outdir + 'dispersion_{0:04d}'.format(kk) + '.png')
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
    lemov.make_movie(imgname, movname, indexsz='04', framerate=2)


# Parameters
# number of sampling points
nn = 800
full3dplot = False
saveims = False
fontsize = 20
tick_fontsize = 16
maindir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/haldane_model/'
outdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/haldane_model/t2_movie/'
# hopping strengths
t1 = 1.
t2arr = np.arange(0, 0.200001, 0.01)
mm = 0.

# Create movie and image sequence
movie_varyt2(t1, t2arr, mm, nn, maindir, outdir, fontsize=fontsize, tick_fontsize=tick_fontsize, saveims=saveims)

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
