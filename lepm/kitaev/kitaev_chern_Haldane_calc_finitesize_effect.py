import numpy as np
import numpy.linalg as la
import lepm.lattice_elasticity as le
import matplotlib.pyplot as plt
import make_lattice as makeL
import copy
import argparse
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
import lepm.plotting.colormaps
import lepm.line_segments as lsegs
import lepm.stringformat as sf

'''Calculate Chern number via realspace method Kitaev method for a range of system sizes (or just one).
'''


def deformed_hexcell_to_hexagonal_sidelengths(delta, NH, NV):
    """
    Parameters
    ----------
    delta : float
        opening angle of top corner of hexagon (oriented with vertices aligned vertically)
    NH :
    NV :

    Returns
    ----------
    a1 : float
        bottom side length of hexagonal polygon
    a2 : float
        vertical/tilted side length of hexagonal polygon.
    """
    eta = 0.25 * (4.*np.pi - 2.*delta)
    print 'eta = ', eta
    a1 = 2. * NH * np.sin( delta*0.5 )
    a2 = 2. * NV * np.sin( eta * 0.5 )
    return a1, a2

def deformed_hex_polygon(a1, a2):
    """
    Make an array demarcating a squished hexagonal polygon with sides a1 (base, horizontal) and a2 (sides, vertical).
    """
    hh = np.sqrt(a2 ** 2 - (a1 * 0.5) ** 2)
    # theta2 = 2*np.arcsin(a1*0.5/a2)
    theta2 = np.arctan2( hh, a1*0.5)
    print 'theta2 = ', theta2

    polygon = np.array([[ a1, 0.], [a2*np.cos(theta2), a2*np.sin(theta2)],
                        [ -a2*np.cos(theta2),  a2*np.sin(theta2)], [ -a1, 0.],
                        [ -a2*np.cos(theta2), -a2*np.sin(theta2)],
                        [  a2*np.cos(theta2), -a2*np.sin(theta2)] ])
    return polygon


def divide_hexagon_by_sidelengths(hexagon, eps=1e-8):
    """
    Input hexagon must be oriented counter-clockwise
    For now assuming starts with vertex on x axis --> make this more general later!
    """
    hex = hexagon
    a1 = hex[0,0]
    # hh = np.sqrt(a2**2 - (a1*0.5)**2)
    hh = hex[1,1]
    print 'hh = ', hh
    theta1 = np.arctan2(a1*0.5,hh)*2.
    print 'theta1 = ', theta1
    apoth = np.sqrt( a2**2 * 1.25 - a2**2 * np.cos(theta1) )
    print 'apoth = ', apoth
    nu = np.arccos((a1**2 + apoth**2 - a2**2 *0.25)/(2*a1*apoth))
    print 'nu = ', nu
    poly1 = np.array([ [apoth*np.cos(nu), -apoth*np.sin(nu)], hex[0], hex[1], [0, hh], [0,0] ])
    poly2 = np.array([ [-eps,hh], hex[2], [-a1,0], [-apoth*np.cos(nu), -apoth*np.sin(nu)], [-eps,0] ])
    poly3 = np.array([ [-apoth*np.cos(nu), -apoth*np.sin(nu)], hex[4], hex[5], [apoth*np.cos(nu+eps), -apoth*np.sin(nu+eps)], [0.,-eps] ])

    return poly1, poly2, poly3


def slice_polygon_regions(polygon,alpha,beta, gamma, eps=1e-9, check=False):
    """
    Parameters
    ----------
    polygon : #vertices x 2 float array
        the polygon to slice into three regions along angles alpha, beta, gamma
        Must be counterclockwise points starting with points near the x axis
        Must also not double back on itself, so that polar angle coordinates are monotonically increasing near
        crossings with alpha, beta, gamma.
    alpha : float
        the smallest angle for a slice
    beta : float
        the second largest (smallest) angle for slicing
    gamma : float
        the largest angle for slicing into three regions

    Returns
    ----------
    polygon1 : #vertices x 2 float array
        first slice of polygon
    polygon2 : #vertices x 2 float array
        second slice of polygon
    polygon3 : #vertices x 2 float array
        third slice of polygon
    """
    if len(np.where(np.abs(polygon.ravel()) > eps)[0]) == 0:
        polygon1 = np.array([[0.,0.],[0.,0.],[0.,0.]])
        polygon2 = np.array([[0., 0.], [0., 0.], [0., 0.]])
        polygon3 = np.array([[0., 0.], [0., 0.], [0., 0.]])
    else:
        # Take alpha, beta, gamma wrt 2 pi
        alpha %= (2.*np.pi)
        beta %= (2.* np.pi)
        gamma %= (2.* np.pi)

        # Make a long ray in each direction alpha, beta, gamma
        norm2s = np.max(polygon[:,0]**2 + polygon[:,1]**2)

        # Order linesegs of polygon by the angles of their endpts
        thetas = np.mod(np.arctan2(polygon[:,1], polygon[:,0]), 2.*np.pi)
        reg1a = np.where(thetas > gamma)[0]
        reg1b = np.where(thetas < alpha)[0]
        reg1IND = np.hstack((reg1a,reg1b))
        reg2IND = np.where(np.logical_and(thetas > alpha, thetas < beta))[0]
        reg3IND = np.where(np.logical_and(thetas > beta, thetas < gamma))[0]

        # First do alpha intersection
        # Get intersection points for divisions between lattice regions
        # Modify the thetas of reg1 points to be wrt gamma
        theta1 = np.hstack(( thetas[reg1a] - gamma, thetas[reg1b]+ np.pi*2 - gamma ))
        print 'theta1 = ', theta1
        a1 = polygon[reg1IND[np.argmax(theta1)],:]
        a2 = polygon[reg2IND[np.argmin(thetas[reg2IND])],:]
        b1 = np.array([ 0.0, 0.0 ])
        b2 = norm2s*np.array([ np.cos(alpha), np.sin(alpha) ])
        intr1 = lsegs.intersection_linesegs(a1, a2, b1, b2, thres=1e-6)

        if check:
            print 'intr1 = ', intr1
            plt.plot(polygon[reg3IND, 0], polygon[reg3IND, 1], 'b.')
            plt.plot(polygon[reg2IND, 0], polygon[reg2IND, 1], 'c.')
            plt.plot(polygon[reg1IND, 0], polygon[reg1IND, 1], 'k.')
            plt.plot(np.array([a1[0]]), np.array([a1[1]]), 'go')
            plt.plot(np.array([a2[0]]), np.array([a2[1]]), 'ro')
            plt.plot(intr1[0], intr1[1], 'rx')
            plt.show()

        a1 = polygon[reg2IND[np.argmax(thetas[reg2IND])],:]
        a2 = polygon[reg3IND[np.argmin(thetas[reg3IND])],:]
        b2 = norm2s*np.array([np.cos(beta), np.sin(beta) ])
        intr2 = lsegs.intersection_linesegs(a1, a2, b1, b2, thres=1e-6)

        if check:
            print 'intr2 = ', intr2
            plt.plot(polygon[reg3IND, 0], polygon[reg3IND, 1], 'b.')
            plt.plot(polygon[reg2IND, 0], polygon[reg2IND, 1], 'c.')
            plt.plot(polygon[reg1IND, 0], polygon[reg1IND, 1], 'k.')
            plt.plot(np.array([a1[0]]), np.array([a1[1]]), 'go')
            plt.plot(np.array([a2[0]]), np.array([a2[1]]), 'ro')
            plt.plot(intr2[0], intr2[1], 'rx')
            plt.show()

        a1 = polygon[reg3IND[np.argmax(thetas[reg3IND])],:]
        a2 = polygon[reg1IND[np.argmin(theta1)],:]
        b2 = norm2s*np.array([ np.cos(gamma), np.sin(gamma) ])
        intr3 = lsegs.intersection_linesegs(a1, a2, b1, b2, thres=1e-6)

        if check:
            print 'thetas[reg1IND] = ', thetas[reg1IND]
            print 'np.mod(thetas[reg1IND],gamma) = ', np.mod(thetas[reg1IND], gamma)
            print 'intr3 = ', intr3
            plt.plot(polygon[reg3IND, 0], polygon[reg3IND, 1], 'b.')
            plt.plot(polygon[reg2IND, 0], polygon[reg2IND, 1], 'c.')
            plt.plot(polygon[reg1IND, 0], polygon[reg1IND, 1], 'k.')
            plt.plot(np.array([a1[0]]), np.array([a1[1]]), 'go')
            plt.plot(np.array([a2[0]]), np.array([a2[1]]), 'ro')
            plt.plot(intr3[0], intr3[1], 'rx')
            plt.show()

        normalp = np.sqrt(intr1[0]**2 + intr1[1]**2)
        normbet = np.sqrt(intr2[0]**2 + intr2[1]**2)
        normgam = np.sqrt(intr3[0]**2 + intr3[1]**2)

        intgamma_reg1 = normgam * np.array([np.cos(gamma+eps), np.sin(gamma+eps)])
        intalpha_reg1 = normalp * np.array([np.cos(alpha-eps), np.sin(alpha-eps)])
        intalpha_reg2 = normalp * np.array([np.cos(alpha+eps), np.sin(alpha+eps)])
        intbeta_reg2 = normbet * np.array([np.cos(beta-eps), np.sin(beta-eps)])
        intbeta_reg3 = normbet * np.array([np.cos(beta+eps), np.sin(beta+eps)])
        intgamma_reg3 = normgam * np.array([np.cos(gamma-eps), np.sin(gamma-eps)])

        polygon1 = np.vstack((np.array([0.0,0.0]), intgamma_reg1, polygon[reg1a], polygon[reg1b], intalpha_reg1))
        polygon2 = np.vstack((np.array([0.0,0.0]), intalpha_reg2, polygon[reg2IND], intbeta_reg2))
        polygon3 = np.vstack((np.array([0.0,0.0]), intbeta_reg3 , polygon[reg3IND], intgamma_reg3))

    return polygon1, polygon2, polygon3


lepm.plotting.colormaps.register_colormaps()

parser = argparse.ArgumentParser(description='Specify parameters for computing Chern number in realspace Haldane model.')
parser.add_argument('datehourmin', type=str, nargs='?',
                       help='Name of simulation to build, in format YYYYMMDD-HHMM',
                       default='check_string_for_empty')
parser.add_argument('-seriesdir','--seriesdir',help='Name for the directory in which to store the calculation datadir', type=str, default='kitaev_chern_Haldane_finsize')    
parser.add_argument('-periodic', '--periodicBC', help='Enforce periodic boundary conditions', action='store_true')
parser.add_argument('-skip_ims', '--skip_ims', help='Do not save DOS or region images', action='store_true')
parser.add_argument('-skip_DOS_ims','--skip_DOS_ims', help='Skip plotting and saving Density Of States results', action='store_true')
parser.add_argument('-N', '--N', help='Mesh width AND height, in number of lattice spacings. Specify a series of values by using / between vals (ex 3/4/5/6)', type=str, default='5')
parser.add_argument('-epsilon','--epsilon', help='Magnitude of next nearest neighbor coupling', type=float, default=0.1)
parser.add_argument('-ksize','--ksize_frac', help='Fraction of system size to include in kitaev calculation. Specify a series of values if desired', type=str, default='0.75')
parser.add_argument('-check','--check', help='Output intermediate steps from calculation', action='store_true')
parser.add_argument('-delta','--delta_lattice', help='for hexagonal lattice, measure of deformation in radians/pi', type=str, default='0.667')

# Kitaev params
parser.add_argument('-poly_offset','--poly_offset', help='Offset for the center of the polygon', type=str, default='none')
parser.add_argument('-polyT','--polyT', help='Transpose polygon', action='store_true')

# GEOMETRY
parser.add_argument('-shape','--shape', help='Shape of the overall mesh geometry', type=str, default='hexagon')
args = parser.parse_args()

maindir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/experiments/'
FSFS = 12

# Get timestamp for this calculation
timestr, newdatadir, LOADPARAMS, datadir = le.determine_simulation_timestamp(args.datehourmin,maindir)
print 'datadir = ', datadir
if not datadir:
    datadir = maindir+args.seriesdir+'/'+timestr+'/'

# Series of sizes to run is N array (Narr)
Narr = sf.string_sequence_to_numpy_array(args.N, dtype = int)
ksize_arr = sf.string_sequence_to_numpy_array(args.ksize_frac, dtype = float)

print 'Narr = ', Narr
print 'type(Narr) = ', type(Narr)
sysV = np.zeros(len(Narr)*len(ksize_arr), dtype=int)
Nreg1V = np.zeros(len(Narr)*len(ksize_arr), dtype=int)
N_V = np.zeros(len(Narr)*len(ksize_arr), dtype=int)
nuV = np.zeros(len(Narr)*len(ksize_arr), dtype=complex)
ksize_V = np.zeros(len(Narr)*len(ksize_arr), dtype=float)
ksys_sizeV = np.zeros_like(ksize_V)

print 'np.shape(nuV) = ', np.shape(nuV)

# parameters
periodicBC = args.periodicBC
epsilon = args.epsilon
shape = args.shape
save_ims = not args.skip_ims

regalph = np.pi*(11./6.)
regbeta = np.pi*(7./6.)
reggamma = np.pi*0.5
if args.poly_offset != 'none':
    if '/' in args.poly_offset:
        splitpo = args.poly_offset.split('/')
    else:
        splitpo = args.poly_offset.split('_')
    print 'split_po = ', splitpo
    poly_offset = np.array([float(splitpo[0]), float(splitpo[1])])
else:
    poly_offset = np.array([0., 0.])
    
# Use each value of N
for jj in range(len(Narr)):
    N = Narr[jj]
    dataNdir = datadir+'N_{0:03d}'.format(N)+'/'
    le.ensure_dir(dataNdir)
    NH = N
    NV = N
    Ns = min(NH, NV)
    
    # Make Lattice
    xy, NL, KL, BL, LVUC, LV, UC, PVxydict, PVx, PVy, lattice_exten = \
        makeL.generate_honeycomb_lattice(shape, NH, NV, delta=np.pi*120./180., phi=0., eta=0., rot=0.,
                                         periodicBC=periodicBC)
    
    # Save system size (NP)s
    sysV[jj] = len(xy)
    NP = len(xy)
    
    # Make Dynamical Matrix
    print 'Saving dynamical matrix...'
    eigvect, eigval, DM = le.save_normal_modes_haldane(dataNdir, xy, NL, KL, BL, epsilon=epsilon,
                                                       save_ims=not args.skip_DOS_ims, PVx=PVx, PVy=PVy)
    
    U = eigvect.transpose()
    U1 = la.inv(U)
    D = np.zeros((len(U), len(U)), dtype=complex)
    
    for ii in range(len(eigval)):
        ev = eigval[ii]
        D[ii, ii] = ev
        
    M = copy.deepcopy(D)
    M[M.real < 0] = 0
    M[M.real > 0] = 1        
    P = np.dot(U, np.dot(M, U1))

    # CHECK the matrices
    le.plot_complex_matrix(P, name='Projector', outpath=dataNdir + 'P_matrix.png', show=False, close=True)
    if args.check:
        le.plot_complex_matrix(D, name='Eigenvalue Matrix', outpath=dataNdir + 'D_matrix.png', show=False, close=True)
        le.plot_complex_matrix(M, name='Cut-Off Eigval Matrix', outpath=dataNdir + 'M_matrix.png', show=False,
                               close=True)
        # P is hermitian
        le.plot_complex_matrix(P - P.transpose().conjugate(), name=r'$P -P^{dagger}$',
                               outpath=dataNdir + 'PminusPdagger_matrix.png', show=False,
                               close=True)
        # Check that P_ij is correct by transforming an eigenvector to itself or to zero
        inds2proj = [1, round(len(eigval) * 0.5), round(len(eigval) * 0.96)]
        le.plot_projection_complex_vects(P, eigvect, eigval, inds2proj, fig='auto',
                                         outpath=dataNdir + 'complex_vects_projection.png', show=False, close=True)

    # for each ksize_frac, perform sum
    for kk in range(len(ksize_arr)):
        ksize_frac = ksize_arr[kk]
        print 'ksize_frac = ', ksize_frac

        # Divide plane into three, A->B->C counterclockwise
        teps = 1e-7  # small offset of region boundaries
        if args.shape == 'hexagon':
            delta = np.pi * float(args.delta_lattice)
            a1, a2 = deformed_hexcell_to_hexagonal_sidelengths(delta, ksize_frac * NH.astype(float),
                                                               ksize_frac * NV.astype(float))

            hexagon = deformed_hex_polygon(a1, a2)
            polygon1, polygon2, polygon3 = divide_hexagon_by_sidelengths(hexagon, eps=1e-8)

        elif shape == 'square':
            ksize = ksize_frac * NH.astype(float)
            ksizeH = ksize
            ksizeV = ksize
            apH = ksizeH * 0.5
            apV = ksizeV * 0.5
            angleL = apH / np.cos(regalph)  # length of rays that come out at angles
            polygon1 = np.array([[0., 0.],
                                 [apH, angleL * np.sin(regalph)],
                                 [apH, apV],
                                 [0., apV]])
            polygon2 = np.array([[0., 0.],
                                 [np.cos(np.pi * 0.5 + teps), apV],
                                 [-apH, apV],
                                 [-apH, angleL * np.sin(regbeta)]])
            polygon3 = np.array([[0., 0.],
                                 [-apH, angleL * np.sin(regbeta + teps)],
                                 [-apH, -apV],
                                 [apH, -apV],
                                 [apH, angleL * np.sin(regalph - teps)]])

        elif shape == 'circle':
            ksize = float(ksize_frac) * float(NH)
            tt = np.linspace(0.0, 2. * np.pi, 100)
            print 'tt = ', tt
            print 'np.dstack((np.cos(tt), np.sin(tt))) = ', np.dstack((np.cos(tt), np.sin(tt)))
            print 'ksize = ', ksize
            circlepoly = ksize * np.dstack((np.cos(tt), np.sin(tt)))[0]
            polygon1, polygon2, polygon3 = slice_polygon_regions(circlepoly, reggamma, regbeta, regalph)

        else:
            RuntimeError('This shape is not yet supported!')

        if args.polyT:
            polygon1 = np.fliplr(polygon1)
            polygon2_tmp = np.fliplr(polygon3)
            polygon3 = np.fliplr(polygon2)
            polygon2 = polygon2_tmp
            print 'polygon1 = ', polygon1

        if args.poly_offset != 'none':
            if '/' in args.poly_offset:
                splitpo = args.poly_offset.split('/')
            else:
                splitpo = args.poly_offset.split('_')
            print 'split_po = ', splitpo
            poly_offset = np.array([float(splitpo[0]), float(splitpo[1])])
            polygon1 += poly_offset
            polygon2 += poly_offset
            polygon3 += poly_offset

        #####################
        reg1 = le.inds_in_polygon(xy, polygon1)
        reg2 = le.inds_in_polygon(xy, polygon2)
        reg3 = le.inds_in_polygon(xy, polygon3)

        # CHECK
        if args.check:
            h = np.zeros((len(P), len(P), len(P)), dtype=complex)
            for j in range(len(P)):
                for k in range(len(P)):
                    for l in range(len(P)):
                        h[j, k, l] = P[j, k] * P[k, l] * P[l, j] - P[j, l] * P[l, k] * P[k, j]

            h *= 12*np.pi*1j

            print 'Summing up h values using the h object (very slow, for checking)...'
            dmyi = 0
            nu = 0. + 0. * 1j
            nu_sum12 = np.zeros((len(reg1),len(reg2)),dtype='complex')
            print 'shape(nu_sum12) = ', np.shape(nu_sum12)
            nu_sum1 = np.zeros(len(reg1),dtype='complex')
            nu_sum2 = np.zeros(len(reg2),dtype='complex')
            nu_sum3 = np.zeros(len(reg3),dtype='complex')
            nu_last12 = 0.
            nu_last1 = 0.
            jind = 0
            kind = 0
            for j in reg1:
                print 'jind = ', jind, ' of ', len(reg1)
                kind = 0
                for k in reg2:
                    for l in reg3:
                        # print 'jkl = ', j, ',', k, ',', l
                        nu += h[j,k,l]
                        # nu += 12 * np.pi * 1j * (P[j, k] * P[k, l] * P[l, j] - P[j, l] * P[l, k] * P[k, j])
                        dmyi += 1
                    nu_sum12[jind,kind] = nu - nu_last12
                    nu_last12 = nu
                    kind += 1
                nu_sum1[jind] = nu - nu_last1
                nu_last1 = nu
                jind += 1


            print 'nu = ', nu

            ksstr = '_ksize' + str(ksize_frac).replace('.', 'p')

            # Display partial sums
            le.plot_complex_matrix(nu_sum12,name='$\sum_l h_{jkl}$ (regA,B)', outpath=dataNdir+'sum_l_hjkl'+ksstr+'.png',
                                   show=False,close=True)
            plt.plot(np.arange(len(nu_sum1)), nu_sum1)
            plt.xlabel('arb. index of region A particles')
            plt.ylabel(r'$\nu$')
            plt.title('$\sum_{kl} h_{jkl}$ Each sum separate (leaving regA inds)')
            plt.savefig(dataNdir+'summing_sequence_N'+str(N)+ksstr+'.pdf')
            plt.clf()

            print 'Performing other partial sums...'
            nu_last2 = 0.
            jind = 0
            kind = 0
            dmyi = 0
            nu = 0. + 0.*1j
            for j in reg2:
                for k in reg3:
                    for l in reg1:
                        nu += h[j,k,l]
                        #nu += 12*np.pi*1j*(P[j,k]*P[k,l]*P[l,j] - P[j,l]*P[l,k]*P[k,j])
                        dmyi +=1
                nu_sum2[jind] = nu - nu_last2
                nu_last2 = nu
                jind += 1

            print 'Performing last other partial sums...'
            nu_last3 = 0.
            jind = 0
            dmyi = 0
            nu = 0. + 0.*1j
            for j in reg3:
                for k in reg1:
                    for l in reg2:
                        nu += h[j,k,l]
                        #nu += 12*np.pi*1j*(P[j,k]*P[k,l]*P[l,j] - P[j,l]*P[l,k]*P[k,j])
                        dmyi +=1
                nu_sum3[jind] = nu - nu_last3
                nu_last3 = nu
                jind += 1

            # Display partial sums spatially
            print 'Plotting partial sums spatially...'
            pos1 = np.where(np.real(nu_sum1) > 0)[0]
            pos2 = np.where(np.real(nu_sum2) > 0)[0]
            pos3 = np.where(np.real(nu_sum3) > 0)[0]
            neg1 = np.where(np.real(nu_sum1) < 0)[0]
            neg2 = np.where(np.real(nu_sum2) < 0)[0]
            neg3 = np.where(np.real(nu_sum3) < 0)[0]
            plt.scatter(xy[reg1, 0][pos1], xy[reg1, 1][pos1], s=np.abs(nu_sum1[pos1])*400, facecolors='g',
                        edgecolors=None, marker='o', label='reg1')
            plt.scatter(xy[reg1, 0][neg1], xy[reg1, 1][neg1], s=np.abs(nu_sum1[neg1])*400, facecolors='r',
                        edgecolors=None, marker='o')
            plt.scatter(xy[reg2, 0][pos2], xy[reg2, 1][pos2], s=np.abs(nu_sum2[pos2])*400, facecolors='g',
                        edgecolors=None, marker='^', label='reg2')
            plt.scatter(xy[reg2, 0][neg2], xy[reg2, 1][neg2], s=np.abs(nu_sum2[neg2])*400, facecolors='r',
                        edgecolors=None, marker='^')
            plt.scatter(xy[reg3, 0][pos3], xy[reg3, 1][pos3], s=np.abs(nu_sum3[pos3])*400, facecolors='g',
                        edgecolors=None, marker='s', label='reg3')
            plt.scatter(xy[reg3, 0][neg3], xy[reg3, 1][neg3], s=np.abs(nu_sum3[neg3])*400, facecolors='r',
                        edgecolors=None, marker='s')
            plt.title('$\sum_{kl} h_{jkl}$ Each sum separate')
            # plt.legend(loc='best')
            patchList = []
            patchList.append(patches.Polygon(polygon1, color='r'))
            patchList.append(patches.Polygon(polygon2, color='g'))
            patchList.append(patches.Polygon(polygon3, color='b'))
            p = PatchCollection(patchList, cmap=cm.jet, alpha=0.05)
            colors = np.linspace(0, 1, 3)[::-1]
            p.set_array(np.array(colors))
            ax = plt.gca()
            ax.add_collection(p)
            ax.axis('equal')
            plt.savefig(dataNdir + 'spatial_sums_N'+str(N)+ksstr+'.pdf')

        else:
            print 'Summing up h values...'
            dmyi = 0
            nu = 0.+0.*1j
            for j in reg1:
                for k in reg2:
                    for l in reg3:
                        #print 'jkl = ', j, ',', k, ',', l
                        #nu += h[j,k,l]
                        nu += 12*np.pi*1j*(P[j,k]*P[k,l]*P[l,j] - P[j,l]*P[l,k]*P[k,j])
                        dmyi +=1

        print 'nu = ', nu
        nuV[jj] = nu

        nuV[jj * len(ksize_arr) + kk] = nu
        N_V[jj * len(ksize_arr) + kk] = N
        sysV[jj * len(ksize_arr) + kk] = NP
        Nreg1V[jj * len(ksize_arr) + kk] = len(reg1)
        ksize_V[jj * len(ksize_arr) + kk] = ksize_frac
        ksys_sizeV[jj * len(ksize_arr) + kk] = len(reg1) + len(reg2) + len(reg3)

        print 'N = ', N
        print 'ksize_frac = ', ksize_frac
        print 'nuV = ', nuV

        # Save regions
        regions = {'reg1': reg1, 'reg2': reg2, 'reg3': reg3,
                   'polygon1': polygon1, 'polygon2': polygon2, 'polygon3': polygon3,
                   # 'reg1_xy': reg1_xy, 'reg2_xy': reg2_xy, 'reg3_xy': reg3_xy
                   }
        fn = dataNdir + 'params_regs_ksize{0:0.3f}'.format(ksize_frac) + '.txt'
        header = 'Region 1 indices for lattice division, ksize=(' + \
                 '{0:0.5f}'.format(ksize_frac * NH.astype(float)) + ', ' + \
                 '{0:0.5f}'.format(ksize_frac * NV.astype(float)) + ') ' + shape
        le.save_dict(regions, fn, header=header)

        if save_ims:
            plt.clf()
            ax = le.display_lattice_2D(xy, BL, bs='none', close=False, colorz=False, colormap='BlueBlackRed',
                                       bgcolor='#FFFFFF', axis_off=True)
            ax.scatter(xy[reg1, 0], xy[reg1, 1], c='r', s=80, marker='o', alpha=0.3, label='reg1', zorder=100)
            ax.scatter(xy[reg2, 0], xy[reg2, 1], c='g', s=80, marker='^', alpha=0.3, label='reg2', zorder=101)
            ax.scatter(xy[reg3, 0], xy[reg3, 1], c='b', s=80, marker='s', alpha=0.3, label='reg3', zorder=102)
            patchList = []
            patchList.append(patches.Polygon(polygon1, color='r'))
            patchList.append(patches.Polygon(polygon2, color='g'))
            patchList.append(patches.Polygon(polygon3, color='b'))
            p = PatchCollection(patchList, cmap=cm.jet, alpha=0.2, zorder=99)
            colors = np.linspace(0, 1, 3)[::-1]
            p.set_array(np.array(colors))
            ax.add_collection(p)
            plt.legend()
            titlestr = r'Division of lattice: $\nu = ${0:0.3f}'.format(nu.real)
            plt.title(titlestr)
            plt.axis('equal')
            plt.savefig(dataNdir + 'division_lattice_regions_ksize{0:0.3f}'.format(ksize_frac) + '.png')



# Plot and save result
nulims = [min(0,min(nuV)*1.2), max(0, max(nuV)*1.2)]
plt.clf()
plt.plot(N_V,nuV,'bo-')
plt.title('Chern number versus system size', fontsize=FSFS)
if shape == 'hexagon':
    plt.xlabel('Number of hexagons on a side', fontsize=FSFS)
else:
    plt.xlabel('System width', fontsize=FSFS)
plt.ylabel('Chern number', fontsize=FSFS)
plt.xlim([0,max(Narr)*1.2])
plt.ylim(nulims)
plt.savefig(datadir+'chern_finsize')

plt.clf()
# for ii in range(len(Narr)):
#     inds = np.arange(ii*len(ksize_arr),(ii+1)*len(ksize_arr))
#     print 'inds = ', inds
#     plt.plot(ksize_V[inds],nuV[inds],'o-',label='N='+str(sysV[ii*len(ksize_arr)]))
plt.plot(sysV,nuV,'bo-')
plt.title('Chern number versus system size', fontsize=FSFS)
plt.xlabel('System size (# sites)', fontsize=FSFS)
plt.ylabel('Chern number', fontsize=FSFS)
plt.xlim([0,max(sysV)*1.2])
plt.ylim(nulims)
plt.savefig(datadir+'chern_finsize_Nsites')

plt.clf()
for ii in range(len(Narr)):
    inds = np.arange(ii*len(ksize_arr),(ii+1)*len(ksize_arr))
    print 'inds = ', inds
    plt.plot(ksize_V[inds],nuV[inds],'o-',label='N='+str(sysV[ii*len(ksize_arr)]))
plt.title('Chern number versus system size', fontsize=FSFS)
plt.xlabel('Fraction of system size in sum', fontsize=FSFS)
plt.ylabel('Chern number', fontsize=FSFS)
plt.xlim([0,max(ksize_arr)*1.2])
plt.ylim(nulims)
plt.legend(loc='best')
plt.savefig(datadir+'chern_finsize_ksizeSum')

plt.clf()
for ii in range(len(Narr)):
    inds = np.arange(ii*len(ksize_arr),(ii+1)*len(ksize_arr))
    print 'inds = ', inds
    plt.plot(ksys_sizeV[inds],nuV[inds],'o-',label='N='+str(sysV[ii*len(ksize_arr)]))
plt.title('Chern number versus system size', fontsize=FSFS)
plt.xlabel('Fraction of system size in sum', fontsize=FSFS)
plt.ylabel('Chern number', fontsize=FSFS)
plt.xlim([0,max(ksys_sizeV)*1.2])
plt.ylim(nulims)
plt.legend(loc='best')
plt.savefig(datadir+'chern_finsize_NsitesSum')









# Save data as txt
XX = np.dstack((N_V,sysV, Nreg1V, ksize_V,ksys_sizeV,nuV.real))[0]
header = 'N, Nparticles, Nreg1, ksize_frac, ksys_size (note this is 2*NP_summed), nu for Chern calculation: basis='+str(args.basis)+' Omg='+str(args.Omg)+' Omk='+str(args.Omk)
np.savetxt(datadir+'chern_finsize.txt', XX, delimiter= ',', header=header)

# Save parameters as param file
params = {'N' : args.N, 'basis': args.basis, 'Omg': args.Omg, 'Omk': args.Omk, 'ksize_frac': args.ksize_frac,
          'omegac': 0, 'poly_offset': poly_offset, 'polyT': args.polyT}
le.save_dict(params,datadir+'parameters.txt','Parameters for finite size effect chern number calcs')