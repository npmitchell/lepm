import lattice_elasticity as le
import numpy as np
import matplotlib.pyplot as plt
import lattice_class
import argparse
import socket
import plotting.colormaps as lecmaps
import lepm.plotting.science_plot_style as sps

"""
Numerical implementation of NNNangles is found in lattice_class.py.
Here, we compute the NNN angles for the twisted kagome, or other lattices, using analytic methods.
"""

def twisted_kagome_theta_nml(alpha):
    """Return the difference between NN-NNN bond angle and particle-NN bond angle for all particles in unit cell:
    theta_nml = (particle-NN angle - NN-NNN angle)

    Parameters
    ----------
    alpha : 1d float array
        The twist angles in radians
    """
    R12 = - alpha
    L12 = np.pi + alpha
    R23 = np.pi * 2./3. - alpha
    L23 = np.pi * 5./3. + alpha
    R31 = np.pi * 4./3. - alpha
    L31 = np.pi * 1./3. + alpha

    p1_tnml = np.vstack((L12 - L23, L12 - R23, R12 - L23, R12 - R23))
    p2_tnml = np.vstack((L23 - L31, L23 - R31, R23 - L31, R23 - R31))
    p3_tnml = np.vstack((L31 - L12, L31 - R12, R31 - L12, R31 - R12))
    return [p1_tnml, p2_tnml, p3_tnml]


def deformed_kagome_theta_nml(x1, x2, x3, z, check=False):
    """Compute NNN bond angles in the deformed kagome network of Kane & Lubensky 2014

    """
    cc = deformed_kagome_vis(x1, x2, x3, z, outpath=None, check=check)

    R12 = np.arctan2(cc[4, 1] - cc[5, 1], cc[4, 0] - cc[5, 0])
    L12 = np.arctan2(cc[7, 1] - cc[5, 1], cc[7, 0] - cc[5, 0])
    R23 = np.arctan2(cc[3, 1] - cc[4, 1], cc[3, 0] - cc[4, 0])
    L23 = np.arctan2(cc[6, 1] - cc[7, 1], cc[6, 0] - cc[7, 0])
    R31 = np.arctan2(cc[5, 1] - cc[3, 1], cc[5, 0] - cc[3, 0])
    L31 = np.arctan2(cc[5, 1] - cc[6, 1], cc[5, 0] - cc[6, 0])

    p1_tnml = np.vstack((L12 - L23, L12 - R23, R12 - L23, R12 - R23))
    p2_tnml = np.vstack((L23 - L31, L23 - R31, R23 - L31, R23 - R31))
    p3_tnml = np.vstack((L31 - L12, L31 - R12, R31 - L12, R31 - R12))
    return [p1_tnml, p2_tnml, p3_tnml]


def deformed_kagome_vis(x1, x2, x3, z, outpath=None, check=False):
    """Create (and possibly visualize) the deformed kagome network of Kane & Lubensky 2014
    """
    #         0
    #
    # 10  11     1    2
    #
    #    9        3
    #
    #  8   7     5    4
    #
    #         6
    # Bravais primitive unit vecs
    a = np.array([[np.cos(2*np.pi*p/3.), np.sin(2*np.pi*p/3.)] for p in range(3)])
    a1 = a[0]
    a2 = a[1]
    a3 = a[2]
    # make unit cell
    x = np.array([x1, x2, x3])
    y = np.array([z/3. + x[np.mod(i-1, 3)] - x[np.mod(i+1, 3)] for i in [0, 1, 2]])
    s = np.array([x[p]*(a[np.mod(p-1, 3)] - a[np.mod(p+1, 3)]) + y[p]*a[p] for p in range(3)])
    s1 = s[0]
    s2 = s[1]
    d1 = a1/2.+s2
    d2 = a2/2.-s1
    d3 = a3/2.
    # nodes at R (from top to bottom) -- like fig 2a of KaneLubensky2014
    cc = np.array([d1+a2,
                  d3+a1+a2,
                  d2+a1,
                  d1,
                  d3+a1,
                  d2-a2,
                  d1+a3,
                  d3,
                  d2+a3,
                  d1+a2+a3,
                  d3+a2,
                  d2])
    if check or outpath is not None:
        # plot the indices of the particles, to check
        plt.plot(cc[:, 0], cc[:, 1], 'k-')
        plt.plot([cc[-1, 0], cc[0, 0]], [cc[-1, 1], cc[0, 1]], 'k-')
        hex = [11, 1, 3, 5, 7, 9, 11]
        plt.plot(cc[hex, 0], cc[hex, 1], 'b-')
        plt.axis('scaled')
        plt.title(r'$(x_1, x_2, x_3, z) = $(' + '{0:0.3f}'.format(x1) + ', {0:0.3f}'.format(x2) +
                  ', {0:0.3f}'.format(x3) + ', {0:0.3f}'.format(z) + ')', fontsize=20)
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.2, 1.2)
        if outpath is not None:
            plt.savefig(outpath)
            plt.clf()
        else:
            iitmp = 0
            plt.show()
            for xy in cc:
                plt.text(xy[0] + 0.1, xy[1], str(ii))
                iitmp += 1
    return cc


def twisted_kagome_vis(alpha, outpath=None):
    psi = [n*np.pi*(1./3.) + (-1)**n * alpha for n in range(6)]
    b = np.array([[np.cos(psi[i]), np.sin(psi[i])] for i in range(6)])

    # nodes at R (from top to bottom) -- like fig 2a of KaneLubensky2014
    xy = np.array([[0., 0.],
                b[0],
                b[0]+b[1],
                b[0]+b[1]+b[2],
                b[0]+b[1]+b[2]+b[3],
                b[0]+b[1]+b[2]+b[3]+b[4]])
    xy = np.vstack((xy, np.array([0., 0.])))
    plt.plot(xy[:, 0], xy[:, 1], 'k-')
    plt.title(r'$\alpha = $' + '{0:0.2f}'.format(alpha/np.pi) + r'$\pi$')
    plt.axis('scaled')
    plt.xlim(-0.6, 2.2)
    plt.ylim(-0.6, 2.2)
    if outpath is not None:
        plt.savefig(outpath)
        plt.clf()
    else:
        plt.show()


# rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/'
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify time string (timestr) for gyro simulation.')
    parser.add_argument('-LT', '--LatticeTop',
                        help='Lattice topology: linear, hexagonal, triangular, deformed_kagome, hyperuniform, ' +
                        'circlebonds, penroserhombTri',
                        type=str, default='twisted_kagome')
    args = parser.parse_args()

    FSFS = 12

    # Do twisted kagome analytically.
    rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/experiments/NNNangles/' + args.LatticeTop + '/'

    if args.LatticeTop == 'twisted_kagome':
        alpha = np.arange(0, np.pi/3, 0.001)
        tnml = twisted_kagome_theta_nml(alpha)
        print np.shape(tnml)

        colors = lecmaps.husl_palette(n_colors=4, h=0.01, s=0.9, l=0.65)
        labels = [r'$L_{ij} - L_{jk}$', r'$L_{ij} - R_{jk}$', r'$R_{ij} - L_{jk}$', r'$R_{ij} - R_{jk}$']
        print 'colors = ', colors

        # Make figure
        Wfig = 180
        x0frac = 0.1
        y0frac = 0.1
        wsfrac = 0.35
        tspace = 12
        vspace = 20
        hspace = 18
        FSFS = 12
        x0 = round(Wfig * x0frac)
        y0 = round(Wfig * y0frac)
        ws = round(Wfig * wsfrac)
        hs = ws
        Hfig = y0 + 2*hs + vspace + tspace

        fig = sps.figure_in_mm(Wfig, Hfig)
        label_params = dict(size=FSFS, fontweight='normal')
        ax = [sps.axes_in_mm(x0, y0, width, height, label=part, label_params=label_params)
              for x0, y0, width, height, part in (
                [x0, y0 + hs + vspace, ws, hs, ''],
                [x0, y0, ws, hs, ''],
                [x0 + ws + hspace, y0, ws, hs, ''],
                [x0 + ws + hspace, y0 + hs + vspace, ws, hs, '']
              )]

        ind = 0
        for tt in tnml:
            noise = np.random.rand(len(tt), len(alpha))
            print 'noise = ', noise
            for kk in range(len(tt)):
                ax[ind].plot(alpha / np.pi, np.mod(2. * tt[kk], 2. * np.pi) / np.pi, '-',
                             lw=len(tt) - kk, color=colors[kk], label=labels[kk])
                kk += 1
            ax[ind].legend()
            ax[ind].plot(alpha / np.pi, np.ones_like(alpha), 'k--')
            ax[ind].set_xlabel(r'$\alpha/\pi$')
            ax[ind].set_ylabel(r'$2 \theta_{nml}/\pi$')
            ax[ind].set_title('Particle ' + str(ind + 1) + r' $\theta_{nml}$')
            ind += 1

        ax[ind].axis('off')
        plt.savefig(rootdir + args.LatticeTop + '_anglesNNN.png')
        plt.show()

        # Plot some key points in the deformation
        step = np.pi/60.
        for alpha in np.arange(0, np.pi/3. + step, step):
            twisted_kagome_vis(alpha, outpath=rootdir + 'vis_alpha' + le.float2pstr(alpha) + '.png')

    else:
        x1 = np.arange(-0.2, 0.2, 0.001)
        x2 = 0.1
        x3 = 0.1
        z = 0.0
        tnml1 = np.zeros((len(x1), 4), dtype=float)
        tnml2 = np.zeros((len(x1), 4), dtype=float)
        tnml3 = np.zeros((len(x1), 4), dtype=float)
        ii = 0
        for xx in x1:
            tmp = deformed_kagome_theta_nml(xx, x2, x3, z)
            tnml1[ii, :] = tmp[0].ravel()
            tnml2[ii, :] = tmp[1].ravel()
            tnml3[ii, :] = tmp[2].ravel()
            ii += 1

        tnml = [tnml1, tnml2, tnml3]

        colors = lecmaps.husl_palette(n_colors=4, h=0.01, s=0.9, l=0.65)
        labels = [r'$L_{ij} - L_{jk}$', r'$L_{ij} - R_{jk}$', r'$R_{ij} - L_{jk}$', r'$R_{ij} - R_{jk}$']
        print 'colors = ', colors

        # Make figure
        Wfig = 180
        x0frac = 0.1
        y0frac = 0.1
        wsfrac = 0.35
        tspace = 25
        vspace = 20
        hspace = 18
        FSFS = 12
        x0 = round(Wfig * x0frac)
        y0 = round(Wfig * y0frac)
        ws = round(Wfig * wsfrac)
        hs = ws
        Hfig = y0 + 2*hs + vspace + tspace

        fig = sps.figure_in_mm(Wfig, Hfig)
        label_params = dict(size=FSFS, fontweight='normal')
        ax = [sps.axes_in_mm(x0, y0, width, height, label=part, label_params=label_params)
              for x0, y0, width, height, part in (
                [x0, y0 + hs + vspace, ws, hs, ''],
                [x0, y0, ws, hs, ''],
                [x0 + ws + hspace, y0, ws, hs, ''],
                [x0 + ws + hspace, y0 + hs + vspace, ws, hs, '']
              )]

        ind = 0
        for tt in tnml:
            tt = tt.T
            for kk in range(len(tt)):
                ax[ind].plot(x1, np.mod(2. * tt[kk], 2. * np.pi) / np.pi, '-',
                             lw=len(tt) - kk, color=colors[kk], label=labels[kk])
                kk += 1
            ax[ind].legend()
            ax[ind].plot(x1, np.ones_like(x1), 'k--')
            ax[ind].set_xlabel(r'$x_1$', fontsize=FSFS)
            ax[ind].set_ylabel(r'$2 \theta_{nml}/\pi$', fontsize=FSFS)
            ax[ind].set_title('Particle ' + str(ind + 1) + r' $\theta_{nml}$', fontsize=FSFS)
            ind += 1

        ax[ind].axis('off')
        title = r'$(x_1, x_2, x_3, z) = $(' + r'$x_1$,' + '{0:0.3f}'.format(x2) + ', {0:0.3f}'.format(x3) +\
                ', {0:0.3f}'.format(z) + ')'
        ax[0].annotate(title, xy=(0.5, 0.9), xycoords='figure fraction', xytext=(0.5, 0.95),
                       textcoords='figure fraction', arrowprops=dict(),
                       ha='center', va='center', fontsize=14)
        plt.savefig(rootdir + args.LatticeTop + '_anglesNNN_x2' + str(x2) + '_x3' + str(x3) + '_z' + str(z) + '.png')
        plt.show()

        # Plot some key points in the deformation
        outdir = rootdir + 'visualize_x2' + str(x2) + '_x3' + str(x3) + '_z' + str(z) + '/'
        le.ensure_dir(outdir)
        step = 0.2/10.
        ii = 0
        for x1ii in np.arange(np.min(x1), np.max(x1) + step, step):
            deformed_kagome_vis(x1ii, x2, x3, z, outpath=outdir + 'vis_' + '{0:05d}'.format(ii) +
                                                         '_x1' + le.float2pstr(x1ii) +
                                                         '_x2' + str(x2) + '_x3' + str(x3) + '_z' + str(z) + '.png')
            ii += 1
