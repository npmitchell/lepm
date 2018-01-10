import weakref
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import scipy as sp
import pylab as P
from numpy import *
import matplotlib
import lepm.plotting.plotting as leplt
import numpy as np

''' functions for plotting haldane model'''


def construct_eigenvalue_DOS_plot_haldane(xy, fig, dos_ax, eig_ax, eigval, eigvect, en, Ni, Nk, marker_num=0,
                                          color_scheme='default', sub_lattice=-1, PVx=[], PVy=[],
                                          black_t0lines=False, mark_t0=True, title='auto', normalization=1., alpha=0.6,
                                          lw=1, zorder=10):
    """puts together lattice and DOS plots and draws normal mode magitudes as circles on top
    
    Parameters
    ----------
    xy: array 2nx3
        Equilibrium position of the gyroscopes
    fig :
        figure with lattice and DOS drawn
    dos_ax:
        axis for the DOS plot
    eig_ax
        axis for the eigenvalue plot
    eigval : array of dimension 2nx1
        Eigenvalues of matrix for system
    eigvect : array of dimension 2nx2n
        Eigenvectors of matrix for system.
        Eigvect is stored as NModes x NP array, like: mode0_psi0, mode0_psi1, ... / mode1_psi0, ...
    en: int
        Number of the eigenvalue you are plotting
    
    Returns
    ----------
    fig :
        completed figure for normal mode
    
    [scat_fg, p, f_mark] :
        things to be cleared before next normal mode is drawn
    """

    ppu = leplt.get_points_per_unit()
    s = leplt.absolute_sizer()

    # re_eigvals = sum(abs(real(eigval)))
    # im_eigvals = sum(abs(imag(eigval)))

    ev = eigval[en]
    ev1 = ev

    # Show where current eigenvalue is in DOS plot (red line ticking current eigval)
    if dos_ax is not None:
        (f_mark,) = dos_ax.plot([ev.real, ev.real], P.ylim(), '-r')
        plt.sca(dos_ax)

    NP = len(xy)

    im1 = np.imag(ev)
    re1 = np.real(ev)
    P.sca(eig_ax)

    if title == 'auto':
        eig_ax.set_title('Mode %d: $\omega=( %0.6f + %0.6f i)$' % (en, re1, im1))
    elif title is not None and title not in ['', 'none']:
        eig_ax.set_title(title)

    # Preallocate ellipsoid plot vars
    shap = eigvect.shape
    angles_arr = np.zeros(NP)
    major_Ax = np.zeros(NP)

    patch = []
    polygon = []
    colors = np.zeros(NP + 2)
    # x_mag = np.zeros(NP)
    # y_mag = np.zeros(NP)

    x0s = np.zeros(NP)
    y0s = np.zeros(NP)

    mag1 = eigvect[en]

    # Eigvect is stored as NModes x NP*2 array, with x and y components alternating, like:
    # x0, y0, x1, y1, ... xNP, yNP.
    mag1x = np.array([mag1[i] for i in range(NP)])
    mag1y = np.array([mag1[i] for i in range(NP)])

    # Pick a series of times to draw out the ellipsoid
    time_arr = np.arange(81) * 2 * pi / (abs(ev1) * 80)
    exp1 = np.exp(1j * ev1 * time_arr)

    # Normalization for the ellipsoids
    lim_mag1 = max(np.array([np.sqrt(2 * abs(exp1 * mag1x[i]) ** 2) for i in range(len(mag1x))]).flatten())
    mag1x /= lim_mag1
    mag1y /= lim_mag1
    mag1x *= normalization
    mag1y *= normalization

    cw = []
    ccw = []
    lines_1 = []
    for i in range(NP):
        unit = mag1x[i]
        x_disps = 0.5 * (exp1 * unit).real
        y_disps = 0.5 * (exp1 * unit).imag

        x_vals = xy[i, 0] + x_disps
        y_vals = xy[i, 1] + y_disps

        # x_mag[i] = max(x_vals-xy[i,0]).real
        # y_mag[i] = max(y_vals-xy[i,1]).real

        poly_points = array([x_vals, y_vals]).T
        polygon = Polygon(poly_points, True)

        # x0 is the marker_num^th element of x_disps 
        x0 = x_disps[marker_num]
        y0 = y_disps[marker_num]

        x0s[i] = x_vals[marker_num]
        y0s[i] = y_vals[marker_num]

        # These are the black lines protruding from pivot point to current position
        lines_1.append([[xy[i, 0], x_vals[marker_num]], [xy[i, 1], y_vals[marker_num]]])

        mag = sqrt(x0 ** 2 + y0 ** 2)
        if mag > 0:
            anglez = np.arccos(x0 / mag)
        else:
            anglez = 0

        if y0 < 0:
            anglez = 2 * np.pi - anglez

        # testangle = arctan2(y0,x0)
        # print ' x0 - x_disps[0] =', x0-x_disps[marker_num]

        angles_arr[i] = anglez

        # print 'polygon = ', poly_points
        patch.append(polygon)

        # Do Fast Fourier Transform (FFT)
        # ff = abs(fft.fft(x_disps + 1j*y_disps))**2
        # ff_freq = fft.fftfreq(len(x_vals), 1)
        # mm_f = ff_freq[ff == max(ff)][0]

        if color_scheme == 'default':
            colors[i] = anglez
        else:
            if sub_lattice[i] == 0:
                colors[i] = 0
            else:
                colors[i] = pi
                # if mm_f > 0:
                #   colors[i] = 0
                # else:
                #   colors[i] = pi

    colors[NP] = 0
    colors[NP + 1] = 2 * pi

    plt.yticks([])
    plt.xticks([])
    # this is the part that puts a dot a t=0 point
    if mark_t0:
        scat_fg = eig_ax.scatter(x0s[cw], y0s[cw], s=s(.02), c='k')
        scat_fg2 = eig_ax.scatter(x0s[cw], y0s[cw], s=s(.02), c='r')
        scat_fg = [scat_fg, scat_fg2]
    else:
        scat_fg = []

    NP = len(xy)
    try:
        NN = shape(Ni)[1]
    except IndexError:
        NN = 0

    z = np.zeros(NP)

    Rnorm = np.array([x0s, y0s, z]).T

    # Bond Stretches
    inc = 0
    stretches = zeros(3 * len(xy))
    if PVx == [] and PVy == []:
        '''There are no periodic boundaries supplied'''
        for i in range(len(xy)):
            if NN > 0:
                for j, k in zip(Ni[i], Nk[i]):
                    if i < j and abs(k) > 0:
                        n1 = float(linalg.norm(Rnorm[i] - Rnorm[j]))
                        n2 = linalg.norm(xy[i] - xy[j])
                        stretches[inc] = (n1 - n2)
                        inc += 1
    else:
        '''There are periodic boundaries supplied'''
        # get boundary particle indices
        KLabs = np.zeros_like(Nk, dtype='int')
        KLabs[Nk > 0] = 1
        boundary = extract_boundary_from_NL(xy, Ni, KLabs)
        for i in range(len(xy)):
            if NN > 0:
                for j, k in zip(Ni[i], Nk[i]):
                    # if i in boundary and j in boundary:
                    #     col = np.where( Ni[i] == j)[0][0]
                    #     print 'col = ', col
                    #     n1 = float( np.linalg.norm(Rnorm[i]-Rnorm[j]) )
                    #     n2 = np.linalg.norm(R[i] - R[j] )
                    #     stretches[inc] = (n1 - n2)
                    #     inc += 1
                    #     
                    #     #test[inc] = [R[i], np.array([R[j,0]+PVx[i,col], R[j,1] + PVy[i,col], 0])]
                    # else:
                    if i < j and abs(k) > 0:
                        n1 = float(np.linalg.norm(Rnorm[i] - Rnorm[j]))
                        n2 = np.linalg.norm(xy[i] - xy[j])
                        stretches[inc] = (n1 - n2)
                        inc += 1

    stretch = np.array(stretches[0:inc])

    # For particles with neighbors, get list of bonds to draw by stretches
    test = list(np.zeros([inc, 1]))
    inc = 0
    xy = np.array([x0s, y0s, z]).T
    if PVx == [] and PVy == []:
        '''There are no periodic boundaries supplied'''
        for i in range(len(xy)):
            if NN > 0:
                for j, k in zip(Ni[i], Nk[i]):
                    if i < j and abs(k) > 0:
                        test[inc] = [xy[(i, j), 0], xy[(i, j), 1]]
                        inc += 1
    else:
        '''There are periodic boundaries supplied'''
        # get boundary particle indices
        KLabs = np.zeros_like(Nk, dtype='int')
        KLabs[Nk > 0] = 1
        boundary = extract_boundary_from_NL(xy, Ni, KLabs)
        for i in range(len(xy)):
            if NN > 0:
                for j, k in zip(Ni[i], Nk[i]):
                    # if i in boundary and j in boundary:
                    #     col = np.where( Ni[i] == j)[0][0]
                    #     print 'i,j = (', i,j, ')'
                    #     print 'PVx[i,col] = ', PVx[i,col]
                    #     print 'PVy[i,col] = ', PVy[i,col]
                    #     test[inc] = [xy[i], np.array([xy[j,0]+PVx[i,col], xy[j,1] + PVy[i,col], 0])]
                    #     #plt.plot([ xy[i,0], xy[j,0]+PVx[i,col]], [xy[i,1], xy[j,1] + PVy[i,col] ], 'k-')
                    #     #plt.plot(xy[:,0], xy[:,1],'b.')
                    #     #plt.show()
                    #     print 'test = ', test
                    #     inc += 1
                    # else:
                    if i < j and abs(k) > 0:
                        test[inc] = [xy[(i, j), 0], xy[(i, j), 1]]
                        inc += 1

    lines = [zip(x, y) for x, y in test]

    # angles[-1] = 0
    # angles[-2] = 2*pi
    lines_st = LineCollection(lines, array=stretch, cmap='seismic', linewidth=8)
    lines_st.set_clim([-1. * 0.25, 1 * 0.25])
    lines_st.set_zorder(2)

    if black_t0lines:
        lines_12 = [zip(x, y) for x, y in lines_1]
        lines_12_st = LineCollection(lines_12, linewidth=0.8)
        lines_12_st.set_color('k')
        eig_ax.add_collection(lines_12_st)
    else:
        lines_12_st = []

    p = PatchCollection(patch, cmap='hsv', lw=lw, alpha=alpha, zorder=zorder)
    p.set_array(array(colors))
    p.set_clim([0, 2 * pi])
    p.set_zorder(1)

    # eig_ax.add_collection(lines_st)
    eig_ax.add_collection(p)

    eig_ax.set_aspect('equal')
    s = leplt.absolute_sizer()

    # erased ev/(2*pi) here npm 2016
    cw_ccw = [cw, ccw, ev]
    # print cw_ccw[1]

    return fig, [scat_fg, p, f_mark, lines_12_st], cw_ccw


def extract_boundary_from_NL(xy, NL, KL):
    """Extract the boundary of a 2D network (xy,NL,KL).
    
    Parameters
    ----------
    xy : NP x 2 float array
        point set in 2D
    NL : NP x NN int array
        Neighbor list. The ith row contains the indices of xy that are the bonded pts to the ith pt.
        Nonexistent bonds are replaced by zero.
    KL : NP x NN int array
        Connectivity list. The jth column of the ith row ==1 if pt i is bonded to pt NL[i,j].
        The jth column of the ith row ==0 if pt i is not bonded to point NL[i,j].
    
    Returns
    ----------
    boundary : #points on boundary x 1 int array
        indices of points living on boundary of the network
    
    """
    # Initialize the list of boundary indices to be larger than necessary
    bb = np.zeros(2 * len(xy), dtype=int)

    # Clear periodicity information from KL
    KL = np.abs(KL)

    # Start with the rightmost point, which is guaranteed to be 
    # at the convex hull and thus also at the outer edge.
    # Then take the first step to be along the minimum angle bond
    rightIND = np.where(xy[:, 0] == max(xy[:, 0]))[0]
    # If there are more than one rightmost point, choose one
    if rightIND.size > 1:
        rightIND = rightIND[0]
    print 'Found rightmost pt: ', rightIND
    print '  with neighbors: ', NL[rightIND]
    # Grab the true neighbors of this starting point
    print 'np.argwhere(KL[rightIND]) = ', np.argwhere(KL[rightIND]).ravel()
    neighbors = NL[rightIND, np.argwhere(KL[rightIND]).ravel()]
    # Compute the angles of the neighbor bonds 
    angles = np.mod(np.arctan2(xy[neighbors, 1] - xy[rightIND, 1], xy[neighbors, 0] - xy[rightIND, 0]).ravel(),
                    2 * np.pi)
    # print 'angles = ', angles
    # Take the second particle to be the one with the lowest bond angle (will be >= pi/2)
    # print ' angles==min--> ', angles==min(angles)
    nextIND = neighbors[angles == min(angles)][0]
    bb[0] = rightIND

    dmyi = 1
    # as long as we haven't completed the full outer edge/boundary, add nextIND
    while nextIND != rightIND:
        # print '\n nextIND = ', nextIND
        # print 'np.argwhere(KL[nextIND]) = ', np.argwhere(KL[nextIND]).ravel()
        bb[dmyi] = nextIND
        n_tmp = NL[nextIND, np.argwhere(KL[nextIND]).ravel()]
        # Exclude previous boundary particle from the neighbors array, UNLESS IT IS THE ONLY ONE,
        # since its angle with itself is zero!
        if len(n_tmp) == 1:
            '''The bond is a lone bond, not part of a triangle.'''
            neighbors = n_tmp
        else:
            neighbors = np.delete(n_tmp, np.where(n_tmp == bb[dmyi - 1])[0])
        # print 'n_tmp = ', n_tmp
        # print 'neighbors = ', neighbors
        angles = np.mod(np.arctan2(xy[neighbors, 1] - xy[nextIND, 1], xy[neighbors, 0] - xy[nextIND, 0]).ravel() - \
                        np.arctan2(xy[bb[dmyi - 1], 1] - xy[nextIND, 1], xy[bb[dmyi - 1], 0] - xy[nextIND, 0]).ravel(),
                        2 * np.pi)
        # print 'angles = ', angles
        # print ' angles==min--> ', angles==min(angles)
        nextIND = neighbors[angles == min(angles)][0]
        # print 'nextIND = ', nextIND

        # Check
        # plt.plot(xy[:,0],xy[:,1],'b.')
        # XX = np.array([xy[bb[dmyi],0], xy[nextIND,0]])
        # YY = np.array([xy[bb[dmyi],1], xy[nextIND,1]])
        # plt.plot(XX,YY,'r-')
        # for i in range(len(xy)):
        #     plt.text(xy[i,0]+0.2,xy[i,1],str(i))
        # plt.gca().set_aspect('equal')
        # plt.show()

        dmyi += 1

    # Truncate the list of boundary indices
    boundary = bb[0:dmyi]

    return boundary
