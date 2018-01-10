import numpy as np
import lepm.lattice_elasticity as le
import lepm.stringformat as sf
import lepm.line_segments as lsegs
import lepm.stringformat as sf
import numpy.linalg as la
import matplotlib.pyplot as plt
import sys
import scipy.stats as scistat
import copy
try:
    import cPickle as pickle
except ImportError:
    import pickle

"""Supporting functions for gyro_lattice_class.py
"""


def param2meshfnexten_name(param):
    """Convert a parameter (string) name into its possibly-abbreviated form that appears as part of a meshfn_exten (the
    string specifier extension to the pathname of a lattice or gyrolattice).

    Parameters
    ----------
    param : str
        A key for a GyroLattice instance's lp (lattice parameter dictionary)

    Returns
    -------
    mfestr : str
    """
    if param == 'ABDelta':
        mfestr = 'ABd'
    elif param == 'V0_pin_flat':
        mfestr = 'Vfpin'
    else:
        raise RuntimeError('Have not yet supported this parameter in string conversion -- add line for it here.')
    return mfestr


def eigvect2displacements_xy(evii):
    """Convert an eigenvector stored as x,y complex numbers into an array of xy displacements"""
    NP = int(len(evii) * 0.5)
    magx = np.array([evii[2 * i] for i in range(NP)])
    magy = np.array([evii[2 * i + 1] for i in range(NP)])
    displ = np.dstack((magx, magy))[0].real
    return displ


# This version was retired on 12-3-2017, the new version uses inner and outer indices to allow fixed boundaries
# def dynamical_matrix_mass_noinnerouter(mlat):
#     """Compute the dynamical matrix for d^2 u/ dt^2 = D u. This code handles periodic or open boundary conditions
#
#     Parameters
#     ----------
#     mlat
#
#     Returns
#     -------
#     matrix : 2N x 2N float array
#         The dynamical matrix for the positions of masses in the mass-spring network
#     """
#     lat = mlat.lattice
#     NP, NN = lat.NL.shape
#     M1 = np.zeros((2 * NP, 2 * NP))
#     M2 = np.zeros((2 * NP, 2 * NP))
#
#     if 'kpin' in mlat.lp:
#         if np.abs(mlat.lp['kpin']) > 1e-10:
#             add_pinning = True
#         else:
#             add_pinning = False
#     else:
#         add_pinning = False
#
#     m2_shape = M2.shape
#
#     # Unpack periodic boundary vectors
#     if lat.PVx is not None and lat.PVy is not None:
#         PVx = lat.PVx
#         PVy = lat.PVy
#     elif lat.PVxydict:
#         PVx, PVy = le.PVxydict2PVxPVy(lat.PVxydict, lat.NL)
#     else:
#         PVx = np.zeros((NP, NN), dtype=float)
#         PVy = np.zeros((NP, NN), dtype=float)
#
#     for i in range(NP):
#         for nn in range(NN):
#             ni = lat.NL[i, nn]
#             k = np.abs(mlat.kk[i, nn])  # true connection?
#
#             diffx = lat.xy[ni, 0] - lat.xy[i, 0] + PVx[i, nn]
#             diffy = lat.xy[ni, 1] - lat.xy[i, 1] + PVy[i, nn]
#
#             # This is Lisa's original version
#             # rij_mag = np.sqrt(diffx**2+diffy**2)
#             # if k!=0:
#             #     alphaij = np.arccos( diffx /rij_mag)
#             # else: alphaij=0
#             # if diffy<0 :
#             #    alphaij=2*np.pi-alphaij
#             # if lat.NK[i,nn] < 0  : alphaij = (np.pi + alphaij)%(2*np.pi)
#
#             # This is my version (05-28-16)
#             if abs(k) > 0:
#                 alphaij = np.arctan2(diffy, diffx)
#
#             Cos = np.cos(alphaij)
#             Sin = np.sin(alphaij)
#
#             if abs(Cos) < 10E-8:
#                 Cos = 0
#             else:
#                 Cos = Cos
#
#             if abs(Sin) < 10E-8:
#                 Sin = 0
#
#             Cos2 = Cos ** 2
#             Sin2 = Sin ** 2
#             CosSin = Cos * Sin
#
#             # Real equations (x components)
#             massi = mlat.mass[i]
#             if massi == 0 or massi < 0:
#                 raise RuntimeError('Encountered zero or negative mass: mass[' + str(i) + '] = ' + str(massi))
#             M1[2 * i, 2 * i] += k * Cos2 / massi
#             M1[2 * i, 2 * i + 1] += k * CosSin / massi
#             M1[2 * i, 2 * ni] += -k * Cos2 / massi
#             M1[2 * i, 2 * ni + 1] += -k * CosSin / massi
#
#             # Imaginary equations (y components)
#             M1[2 * i + 1, 2 * i] += k * CosSin / massi
#             M1[2 * i + 1, 2 * i + 1] += k * Sin2 / massi
#             M1[2 * i + 1, 2 * ni] += -k * CosSin / massi
#             M1[2 * i + 1, 2 * ni + 1] += -k * Sin2 / massi
#
#             if add_pinning:
#                 # pinning
#                 M2[2 * i, 2 * i] = pin[i]
#                 M2[2 * i + 1, 2 * i + 1] = pin[i]
#
#     matrix = - M1 - M2
#     return matrix


def dynamical_matrix_mass(mlat):
    """Compute the dynamical matrix for d^2 u/ dt^2 = D u. This code handles periodic or open boundary conditions

    Parameters
    ----------
    mlat

    Returns
    -------
    matrix : 2N x 2N float array
        The dynamical matrix for the positions of masses in the mass-spring network
    """
    lat = mlat.lattice
    NP, NN = lat.NL.shape
    M1 = np.zeros((2 * NP, 2 * NP))
    M2 = np.zeros((2 * NP, 2 * NP))

    if 'kpin' in mlat.lp:
        if np.abs(mlat.lp['kpin']) > 1e-10:
            add_pinning = True
        else:
            add_pinning = False
    else:
        add_pinning = False

    m2_shape = M2.shape

    # Unpack periodic boundary vectors
    if lat.PVx is not None and lat.PVy is not None:
        PVx = lat.PVx
        PVy = lat.PVy
    elif lat.PVxydict:
        PVx, PVy = le.PVxydict2PVxPVy(lat.PVxydict, lat.NL)
    else:
        PVx = np.zeros((NP, NN), dtype=float)
        PVy = np.zeros((NP, NN), dtype=float)

    for i in range(NP):
        for nn in range(NN):
            ni = lat.NL[i, nn]
            k = np.abs(mlat.kk[i, nn])  # true connection?

            diffx = lat.xy[ni, 0] - lat.xy[i, 0] + PVx[i, nn]
            diffy = lat.xy[ni, 1] - lat.xy[i, 1] + PVy[i, nn]

            # This is Lisa's original version
            # rij_mag = np.sqrt(diffx**2+diffy**2)
            # if k!=0:
            #     alphaij = np.arccos( diffx /rij_mag)
            # else: alphaij=0
            # if diffy<0 :
            #    alphaij=2*np.pi-alphaij
            # if lat.NK[i,nn] < 0  : alphaij = (np.pi + alphaij)%(2*np.pi)

            # This is my version (05-28-16)
            if abs(k) > 0:
                alphaij = np.arctan2(diffy, diffx)

            Cos = np.cos(alphaij)
            Sin = np.sin(alphaij)

            if abs(Cos) < 10E-8:
                Cos = 0
            else:
                Cos = Cos

            if abs(Sin) < 10E-8:
                Sin = 0

            Cos2 = Cos ** 2
            Sin2 = Sin ** 2
            CosSin = Cos * Sin

            # Real equations (x components)
            massi = mlat.mass[i]
            if massi == 0 or massi < 0:
                raise RuntimeError('Encountered zero or negative mass: mass[' + str(i) + '] = ' + str(massi))
            M1[2 * i, 2 * i] += k * Cos2 / massi
            M1[2 * i, 2 * i + 1] += k * CosSin / massi
            M1[2 * i, 2 * ni] += -k * Cos2 / massi
            M1[2 * i, 2 * ni + 1] += -k * CosSin / massi

            # Imaginary equations (y components)
            M1[2 * i + 1, 2 * i] += k * CosSin / massi
            M1[2 * i + 1, 2 * i + 1] += k * Sin2 / massi
            M1[2 * i + 1, 2 * ni] += -k * CosSin / massi
            M1[2 * i + 1, 2 * ni + 1] += -k * Sin2 / massi

            if add_pinning:
                # pinning
                M2[2 * i, 2 * i] = pin[i]
                M2[2 * i + 1, 2 * i + 1] = pin[i]

    matrix = - M1 - M2
    return matrix


def dynamical_matrix_twistedbc(mlat, eps=1e-10):
    """Compute the dynamical matrix for d^2 u/ dt^2 = D u, with arbitrary phase kx,ky intended to be added to the
    image particles that live across periodic boundary conditions. THis code isn't quite finished, since I need to make
    pvxm and pvym, which are 2np x 2np matrices with the x and y displacements in the right spots for the whole
    dynamical matrix to be build up like:
    for kxi in kx:
        for kyj in ky:
            mat, mpx, mpy, pvxm, pvym = mlatfns.dynamical_matrix_twistedbc(self)
            kxfactor = np.exp(1j * pvxm * kx)
            kyfactor = np.exp(1j * pvym * ky)
            matrix = mat + kxfactor * mpx + kyfactor * mpy
            eigval, eigvect = np.linalg.eig(matrix)
            omegas[ii, jj] = np.sqrt(-eigval)

    Parameters
    ----------
    mlat

    Returns
    -------
    matrix : 2N x 2N float array
        The dynamical matrix for the positions of masses in the mass-spring network
    """
    lat = mlat.lattice
    NP, NN = lat.NL.shape
    M1 = np.zeros((2 * NP, 2 * NP))
    # Instantiate the periodic component to the dynamical matrix
    Mpx = np.zeros((2 * NP, 2 * NP))
    Mpy = np.zeros((2 * NP, 2 * NP))
    M2 = np.zeros((2 * NP, 2 * NP))

    if 'kpin' in mlat.lp:
        if np.abs(mlat.lp['kpin']) > 1e-10:
            add_pinning = True
        else:
            add_pinning = False
    else:
        add_pinning = False

    m2_shape = M2.shape

    # Unpack periodic boundary vectors
    if lat.PVx is not None and lat.PVy is not None:
        PVx = lat.PVx
        PVy = lat.PVy
    elif lat.PVxydict:
        PVx, PVy = le.PVxydict2PVxPVy(lat.PVxydict, lat.NL)
    else:
        PVx = np.zeros((NP, NN), dtype=float)
        PVy = np.zeros((NP, NN), dtype=float)

    for i in range(NP):
        for nn in range(NN):
            ni = lat.NL[i, nn]
            k = np.abs(mlat.kk[i, nn])  # true connection?

            diffx = lat.xy[ni, 0] - lat.xy[i, 0] + PVx[i, nn]
            diffy = lat.xy[ni, 1] - lat.xy[i, 1] + PVy[i, nn]

            if abs(k) > 0:
                alphaij = np.arctan2(diffy, diffx)

            Cos = np.cos(alphaij)
            Sin = np.sin(alphaij)

            if abs(Cos) < 10E-8:
                Cos = 0
            else:
                Cos = Cos

            if abs(Sin) < 10E-8:
                Sin = 0

            Cos2 = Cos ** 2
            Sin2 = Sin ** 2
            CosSin = Cos * Sin

            # Real equations (x components)
            massi = mlat.mass[i]
            if massi == 0 or massi < 0:
                raise RuntimeError('Encountered zero or negative mass: mass[' + str(i) + '] = ' + str(massi))

            # Add components to dynamical matrix
            M1[2 * i, 2 * i] += k * Cos2 / massi
            M1[2 * i, 2 * i + 1] += k * CosSin / massi
            M1[2 * i + 1, 2 * i] += k * CosSin / massi
            M1[2 * i + 1, 2 * i + 1] += k * Sin2 / massi

            if np.abs(PVx[i, nn]) > eps and np.abs(PVy[i, nn]) > eps:
                # Add to the periodic matrix
                # x components
                Mpx[2 * i, 2 * ni] += -k * Cos2 / massi
                Mpy[2 * i, 2 * ni + 1] += -k * CosSin / massi

                # y components
                Mpx[2 * i + 1, 2 * ni] += -k * CosSin / massi
                Mpy[2 * i + 1, 2 * ni + 1] += -k * Sin2 / massi

                # todo: add components to pvxm and pvym here
            else:
                M1[2 * i, 2 * ni] += -k * Cos2 / massi
                M1[2 * i, 2 * ni + 1] += -k * CosSin / massi

                # Imaginary equations (y components)
                M1[2 * i + 1, 2 * ni] += -k * CosSin / massi
                M1[2 * i + 1, 2 * ni + 1] += -k * Sin2 / massi

            if add_pinning:
                # pinning
                M2[2 * i, 2 * i] = pin[i]
                M2[2 * i + 1, 2 * i + 1] = pin[i]

    matrix = - M1 - M2
    return matrix, -Mpx, -Mpy, pvxm, pvym


def kk_spec_gridlines(lat, lp, gridspacing, maxval, strongbond_val, weakbond_val, check=False):
    """Build kk (matrix of spring frequencies) with weakened bonds from a supplied lattice.

    Parameters
    ----------
    lat : lepm.lattice_class.Lattice instance
        The lattice on which to build kk
    gridspacing : float
        The spacing between gridlines for which to weaken bonds
    maxval : float
        The extremal value of x or y coordinate in the network, to use for finding bonds to weaken
    strongbond_val : float
        The spring frequency for the stronger, unaltered bonds
    weakbond_val : float
        The spring frequency for the altered bonds (assumed weaker)
    check : bool
        Display the pattern of strong and weak bonds as blue and red before continuing

    Returns
    -------
    kk : N x max(#NN) float array
        bond frequencies matching the KL and NL arrays, with bonds weakened along gridlines
    """
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
    pvx = lat.PVx
    pvy = lat.PVy
    NL = lat.NL

    if lp['periodicBC']:
        bondsegs = np.array([[xy[b[0], 0],
                              xy[b[0], 1],
                              xy[b[1], 0] + pvx[b[0], np.where(NL[b[0]] == b[1])[0]],
                              xy[b[1], 1] + pvy[b[0], np.where(NL[b[0]] == b[1])[0]]] for b in np.abs(lat.BL)])
    else:
        bondsegs = np.array([[xy[b[0], 0],
                              xy[b[0], 1],
                              xy[b[1], 0],
                              xy[b[1], 1]] for b in np.abs(lat.BL)])  # get crossings

    # print 'gridsegs = ', gridsegs
    does_intersect = lsegs.linesegs_intersect_linesegs(bondsegs, gridsegs)

    if check:
        kk = 0
        xinds = [0, 2]
        yinds = [1, 3]
        aBL = np.abs(lat.BL)
        for bb in bondsegs:
            if does_intersect[kk]:
                plt.plot(bb[xinds], bb[yinds], 'r.-')
            else:
                plt.plot(bb[xinds], bb[yinds], 'b.-')
            kk += 1
        plt.show()

        kk = 0
        for bb in aBL:
            if does_intersect[kk]:
                plt.plot(xy[bb, 0], xy[bb, 1], 'r.-')
            else:
                plt.plot(xy[bb, 0], xy[bb, 1], 'b.-')
            kk += 1
        plt.show()

    # Build kk as it would be constructed without the kkspec specification
    # In a copy of lp, kill the kk specification to avoid getting stuck in a loop
    from lepm.gyro_lattice_class import GyroLattice
    lp_copy = copy.deepcopy(lp)
    lp_copy['kkspec'] = None
    lp_copy['kk'] = strongbond_val
    tmp_glat = GyroLattice(lat, lp_copy)
    kk = copy.deepcopy(tmp_glat.kk)

    # Now weaken the gridline bonds. Note that we take the absolute value of BL to allow for periodic networks
    print 'Altering weak bonds --> ', weakbond_val
    # print 'lat.BL[does_intersect] = ', lat.BL[does_intersect]
    for bond in np.abs(lat.BL[does_intersect]):
        kk[bond[0], np.where(lat.NL[bond[0]] == bond[1])[0]] = weakbond_val
        kk[bond[1], np.where(lat.NL[bond[1]] == bond[0])[0]] = weakbond_val

    # check it
    if check:
        kkv = le.KL2kL(lat.NL, kk, lat.BL)
        test = -np.ones(len(lat.BL[:, 0]), dtype=float)
        test[does_intersect] = weakbond_val
        print 'test = ', test
        le.display_lattice_2D(lat.xy, lat.BL, NL=lat.NL, KL=lat.KL, PVxydict=lat.PVxydict,
                              PVx=lat.PVx, PVy=lat.PVy, bs=test,
                              title='', xlimv=None, ylimv=None, climv=0.1, colorz=True, ptcolor=None, ptsize=10,
                              close=True, colorpoly=False, viewmethod=False, labelinds=False,
                              colormap='seismic', bgcolor='#d9d9d9', axis_off=False, fig=None, ax=None, linewidth=0.0,
                              edgecolors=None, check=True)
        le.display_lattice_2D(lat.xy, lat.BL, NL=lat.NL, KL=lat.KL, PVxydict=lat.PVxydict,
                              PVx=lat.PVx, PVy=lat.PVy, bs=kkv,
                              title='', xlimv=None, ylimv=None, climv=0.1, colorz=True, ptcolor=None, ptsize=10,
                              close=True, colorpoly=False, viewmethod=False, labelinds=False,
                              colormap='seismic', bgcolor='#d9d9d9', axis_off=False, fig=None, ax=None, linewidth=0.0,
                              edgecolors=None, check=True)
    return kk


def build_kk(lattice, lp, kk):
    """Construct kk from supplied kk (may be None or 'none') and lattice parameter dictionary lp

    Parameters
    ----------
    lattice : lattice_class.Lattice() instance
    kk : N x max(#NN) float array or None or 'none'
        spring constants matching the KL and NL arrays, with bonds weakened along gridlines
    lp : dict
        lattice parameter dictionary

    Returns
    -------
    kk :N x max(#NN) float array
        spring constants matching the KL and NL arrays, with bonds weakened along gridlines
    lp_kk : float
        value for key 'kk' to be added to lp
    lp_meshfn_exten : str
        value for key 'meshfn_exten' to be added to lp
    """
    if kk == 'auto' or kk is None:
        # Check if a string specifier for kk is given. If it can be understood, create that bond strength pattern.
        # If it is an unknown string, then kk must be supplied.
        if 'kkspec' in lp:
            if lp['kkspec'] not in ['', 'none', None]:
                if 'gridlines' in lp['kkspec']:
                    # Here, kkspec must be of the form:
                    # 'gridlines{0:0.2f}'.format(gridspacing).replace('.', 'p') + \
                    # 'strong{0:0.3f}'.format(lp['kk']).replace('.', 'p').replace('-', 'n') + \
                    # 'weak{0:0.3f}'.format(args.weakbond_val).replace('.', 'p').replace('-', 'n')
                    spec = lp['kkspec'].split('gridlines')[1]
                    gridspacing = float(spec.split('strong')[0].replace('p', '.'))
                    strong = float(spec.split('strong')[1].split('weak')[0].replace('p', '.').replace('n', '-'))
                    weak = float(spec.split('weak')[1].replace('p', '.').replace('n', '-'))
                    maxval = max(np.max(np.abs(lattice.xy[:, 0])), np.max(np.abs(lattice.xy[:, 1]))) + 1
                    kk = kk_spec_gridlines(lattice, lp, gridspacing, maxval, strong, weak)
                else:
                    raise RuntimeError('kkspec in lp cannot be translated into kk, must supply kk or edit to' +
                                       ' interpret given kkspec')
                lp_meshfn_exten = '_kkspec' + lp['kkspec']
                if (kk == kk[np.nonzero(kk)][0] * lattice.KL).all():
                    # This is just the boring case where kk is not special (all bonds are equal).
                    lp_kk = kk[np.nonzero(kk)][0]
                    if lp_kk != 1.0:
                        lp_meshfn_exten = '_kk' + sf.float2pstr(lp['kk'])
                    else:
                        lp_meshfn_exten = ''
                elif (kk != lp['kk'] * np.abs(lattice.KL)).any():
                    lp_kk = -5000
                else:
                    lp_kk = lp['kk']
                done = True
            else:
                # kkspec is in lp, but it is None or 'none'. Try a different method to obtain kk.
                done = False
        else:
            # kkspec is not in lp. Try a different method to obtain kk.
            done = False

        if not done:
            if 'kk' in lp:
                print 'gyro_lattice_class: using kk from lp...'
                kk = lp['kk'] * np.abs(lattice.KL)
                lp_kk = lp['kk']
                if lp_kk != 1.0:
                    lp_meshfn_exten = '_kk' + sf.float2pstr(lp['kk'])
                else:
                    lp_meshfn_exten = ''
            else:
                print 'giving kk the default value of 1s...'
                kk = 1.0 * np.abs(lattice.KL)
                lp_kk = 1.0
                lp_meshfn_exten = ''
    else:
        # This is the case where kk is specified. Pass it along to output and discern what to give for lp_meshfn_exten
        # Output kk <-- input kk
        # Use given kk to define lp_kk (for lp['kk']) and lp[meshfn_exten].
        # If kk is a key in lp, correct it if it does not match supplied kk
        if 'kk' in lp:
            if (kk == lp['kk'] * np.abs(lattice.KL)).any():
                # This is the case that kk = kk * np.abs(KL). Give a nontrivial meshfn_exten if kk != 1.0
                lp_kk = lp['kk']
                if lp_kk != 1.0:
                    lp_meshfn_exten = '_kk' + sf.float2pstr(lp['kk'])
                else:
                    lp_meshfn_exten = ''
            else:
                # kk given is either some constant times KL, or something complicated
                # If it is a constant, discern that constant and update lp
                kinds = np.nonzero(kk)
                if len(kinds[0]) > 0:
                    # There are some nonzero elements in kk. Check if all the same
                    kkravel = kk.ravel()
                    KLravel = lattice.KL.ravel()
                    if (kkravel[np.where(abs(KLravel))] == kkravel[np.where(abs(KLravel))[0]]).all():
                        print 'Updating lp[kk] to reflect specified kk, since kk = constant * np.abs(KL)...'
                        lp_kk = kkravel[np.where(abs(KLravel))[0]]
                    else:
                        # kk is something complicated, so tell meshfn_exten that kk is specified.
                        lp_meshfn_exten = '_kkspec'
                        if 'kkspec' in lp:
                            lp_meshfn_exten = '_kkspec' + lp['kkspec']
                        lp_kk = -5000

                else:
                    lp_kk = 0.0
                    lp_meshfn_exten = '_kk0p00'
        else:
            # Check if the values of all elements are identical
            kinds = np.nonzero(kk)
            if len(kinds[0]) > 0:
                # There are some nonzero elements in kk. Check if all the same
                value = kk[kinds[0][0], kinds[1][0]]
                if (kk[kinds] == value).all():
                    lp_kk = value
                else:
                    lp_kk = -5000
            else:
                lp_kk = 0.0
                lp_meshfn_exten = '_kk0p00'

    return kk, lp_kk, lp_meshfn_exten


def build_kkspec_list(gridspacing, strongbond_val, weakbond_vals):
    """Build up kkspec according to the format:
    'gridlines{0:0.2f}'.format(gridspacing).replace('.', 'p') + \
    'strong{0:0.3f}'.format(lp['kk']).replace('.', 'p').replace('-', 'n') + \
    'weak{0:0.3f}'.format(args.weakbond_val).replace('.', 'p').replace('-', 'n')


    Parameters
    ----------
    gridspacing : float
        Spatial interval at which to weaken bonds
    strong_bondval : float
        value of strong spring constant
    weak_bondvals : string, float, or numpy array
        values for the spring frequencies for the altered (weakened) springs

    Returns
    -------
    kkspecList : list of strings
        list of string specifiers for a MassLattice's kk
    """
    kkspecList = []
    if isinstance(weakbond_vals, str):
        if '/' in weakbond_vals or ':' in weakbond_vals:
            weakarr = sf.string_sequence_to_numpy_array(weakbond_vals, dtype=float)
            for weak in weakarr:
                kkspecList.append('gridlines{0:0.2f}'.format(gridspacing).replace('.', 'p') +
                                  'strong{0:0.3f}'.format(strongbond_val).replace('.', 'p').replace('-', 'n') +
                                  'weak{0:0.3f}'.format(weak).replace('.', 'p').replace('-', 'n'))
        else:
            kkspecList.append('gridlines{0:0.2f}'.format(gridspacing).replace('.', 'p') +
                              'strong{0:0.3f}'.format(strongbond_val).replace('.', 'p').replace('-', 'n') +
                              'weak{0:0.3f}'.format(
                                   float(sf.str2float(weakbond_vals, ndigits=3))).replace('.', 'p').replace('-', 'n'))
    elif isinstance(weakbond_vals, np.ndarray):
        for weak in weakarr:
            kkspecList.append('gridlines{0:0.2f}'.format(gridspacing).replace('.', 'p') +
                              'strong{0:0.3f}'.format(strongbond_val).replace('.', 'p').replace('-', 'n') +
                              'weak{0:0.3f}'.format(weak).replace('.', 'p').replace('-', 'n'))

    elif isinstance(weakbond_vals, float):
        kkspecList.append('gridlines{0:0.2f}'.format(gridspacing).replace('.', 'p') +
                          'strong{0:0.3f}'.format(strongbond_val).replace('.', 'p').replace('-', 'n') +
                          'weak{0:0.3f}'.format(weakbond_vals).replace('.', 'p').replace('-', 'n'))
    return kkspecList


def ascribe_absites(lat):
    """Alternatingly define A-sites and B-sites (ABsites) for particles (typically with three nearest neighbors)

    Parameters
    ----------
    lat : Lattice class instance
        the lattice for which to find ABsites
    """
    thres = 1e-7
    left = np.where(lat.xy[:, 0] + thres > np.min(lat.xy[:, 0]))[0]
    botleft = left[np.argmin(lat.xy[left, 1])]
    asites = np.zeros(len(lat.xy), dtype=bool)
    bsites = np.zeros(len(lat.xy), dtype=bool)
    # done monitors which sites' neighbors have been flipped
    done = np.zeros(len(lat.xy), dtype=bool)
    asites[botleft] = True
    xy = lat.xy
    NL = lat.NL
    KL = lat.KL
    dmyi = 0
    while not done.all():
        # ascribe neighbors of a to be b's. Ascribe neighbors of b's to be a's
        todo = np.where(np.logical_and(asites, ~done))[0]
        for ii in todo:
            flip = np.setdiff1d(np.setdiff1d(NL[ii][np.where(KL[ii])[0]], np.where(asites)[0]),
                                np.where(bsites)[0])
            # print 'flip = ', flip
            bsites[flip] = True
        done[todo] = True

        print 'todo = ', todo
        todo = np.where(np.logical_and(bsites, ~done))[0]
        for ii in todo:
            flip = np.setdiff1d(np.setdiff1d(NL[ii][np.where(KL[ii])[0]], np.where(asites)[0]),
                                np.where(bsites)[0])
            # print 'flip = ', flip
            asites[flip] = True
        done[todo] = True

        if lat.lp['check']:
            plt.scatter(xy[:, 0], xy[:, 1], c='k')
            plt.scatter(xy[asites, 0], xy[asites, 1], c='r')
            plt.scatter(xy[bsites, 0], xy[bsites, 1], c='g')
            plt.pause(0.1)
            plt.clf()
            dmyi += 1

    return asites, bsites


def delta_eo(eta, omega, omegaj):
    """This is delta_eta(omega), the sharp Lorentzian used in diffusivity calculations"""
    return eta / (np.pi * (omega - omegaj)**2 + eta**2)


def lorentzian(eps, omega, omegaj):
    """This is a sharp Lorentzian used in LDOS calculations

    Parameters
    ----------
    eps : float
        A small value specifying the height and narrowness of the Lorentzian (the taller, the skinnier)
    omega : float
        The central value of symmetry for the Lorentzian
    omegaj : float or float array
        Values at which to evaluate the Lorentzian

    Returns
    -------
    a Lorentzian evaluated at the value(s) omegaj
    """
    return eps / (np.pi * ((omega - omegaj)**2 + eps**2))


def calc_ldos(eigval, eigvect, eps=5.0):
    """Compute the local density of states rho_i(omega) = sum_n delta(omega-omega_n) |<i|n>|^2
    Returns
    -------
    ldos : NP x len(eigval) float array
        row i is the mod(i, 2)th particle, column j is the jth eigenvalue
    """
    # METHOD 1
    eta = eps * np.median(np.abs(np.diff(eigval)))
    ldos_tmp = np.zeros((len(eigval), int(0.5 * len(eigval))))
    ii = 0
    inds = np.arange(int(0.5 * len(eigval)), len(eigval))
    imeval = np.abs(eigval)

    # Check normalization of states: np.norm(a) = np.sqrt(np.sum(np.abs(a)**2)) = 1.
    # dotp = np.zeros(len(inds), dtype=complex)
    # ii = 0
    # for row in eigvect[inds]:
    #     dotp[ii] = la.norm(row)
    #     ii += 1
    # print 'dotp = ', dotp
    # plt.plot(np.real(eigval[inds]), np.real(dotp), 'k.-')
    # plt.plot(np.real(eigval[inds]), np.imag(dotp), 'r.-')
    # plt.show()

    for ev in imeval:
        # consider the mod(ii, 2) particle/site
        # Sum over nn, taking iith amplitude in each nnth eigenmode, weighted by delta(omega-omega_n)
        # |<ii|nn>|^2 = |nn>[ii]**2
        print 'ev = ', ev
        jj = 0
        for omega in imeval[inds]:
            # rho_i(omega) = sum_n delta(omega-omega_n) |<i|n>|^2

            # If desired, we can use the full eigenvalue, but only half is needed.
            # lrz = lorentzian(eta, omega, imeval)
            # plt.plot(imeval, lrz, '-')
            # ldos_tmp[ii, jj] = np.sum(lrz * (eigvect[:, ii])**2)

            lrz = lorentzian(eta, omega, imeval[inds])
            ldos_tmp[ii, jj] = np.sum(lrz * (np.abs(eigvect[inds, ii]))**2)
            jj += 1
        ii += 1
        # plt.show()
        # sys.exit()

    # METHOD 2
    # In the limit that the Lorentzian becomes a delta function, we get
    # ldos[ii, jj] = eigvect[jj, ii] = eigvect.T
    # ldos = eigvect[inds].T

    ldos = np.array([np.sqrt(ldos_tmp[2*ii] ** 2 + ldos_tmp[2*ii + 1] ** 2) for ii in range(len(inds))])

    return ldos


def fit_eigvect_to_exponential(xy, eigval, eigvect, cutoffd=None, check=False):
    """Fit the excitations of a mass-spring lattice to exponential decay: finite system, non-PBCs.
    For PBCs, use fit_eigvect_to_exponential_periodic()

    Parameters
    ----------
    xy : NP x 2 float array
        positions of particles in 2D
    eigval : 2*NP x 1 complex array
        eigenvalues of the system
    eigvect : 2*NP x 2*NP complex array
        eigenvectors of the system
    cutoffd : float
        cutoff distance for fitting exponential
    check : bool
        show intermediate steps

    Returns
    -------
    fits : NP x 7 float array (ie, int(len(eigval)*0.5) x 7 float array)
        fit details: x_center, y_center, A, K, uncertainty_A, covariance_AK, uncertainty_K
    """
    raise RuntimeError("I think this does not currently work --> check that the right eigvects are being considered")
    # convert eigvect to magnitude of eigenvectors
    # Contract x and y component of each evect so that len(eigvect[ii]) = NP
    magevec = np.zeros((len(eigval), len(eigval)), dtype=float)
    jj = 0
    for row in eigvect:
        magevec[jj] = np.array([np.sqrt(np.abs(row[2*ii])**2 + np.abs(row[2*ii + 1])**2)
                                for ii in np.arange(0, int(len(eigvect) * 0.5))])
        jj += 1

    # find COM for each eigvect
    fits = np.zeros((len(eigval), 7), dtype=float)
    for ii in range(len(eigval)):
        if ii % 100 == 1:
            print 'glfns: eval #', str(ii), '/', len(eigval)
        com = le.center_of_mass(xy, magevec[ii])
        fits[ii, 0:2] = com

        # Get minimum distance for each particle
        dist = np.sqrt((xy[:, 0] - com[0])**2 + (xy[:, 1] - com[1])**2)

        # Sort particles by distance from com
        inds = np.argsort(dist)
        # print 'dist[inds] = ', dist[inds]
        if cutoffd is not None:
            inds = inds[dist[inds] < cutoffd]
        mags = magevec[ii][inds]
        dists = dist[inds]

        # Check if exponential is possible
        # kstest_exponential(magevec[ii], dist, bins=int(np.max(dist)))
        # possibly_exp = possibly_exponential(dists, mags)

        # if possibly_exp:
        # Fit to an exponential y=A*exp(K*t) using linear method.
        # METHOD 1
        # bin_means, bin_edges, bin_number = scistat.binned_statistic(dists, mags, bins=int(cutoffd))
        # A, K, cov = fit_exp_linear(binc, bin_means)
        A, K, cov = fit_exp_linear(dists, mags)
        fits[ii, 2] = A
        fits[ii, 3] = K
        fits[ii, 4] = cov[0, 0]
        fits[ii, 5] = cov[0, 1]
        fits[ii, 6] = cov[1, 1]

        # check the result
        if check:
            plt.plot(dists, mags, 'b.-')
            Astr = '  A = {0:0.5f}'.format(A) + r'$\pm$' + '{0:0.5f}'.format(fits[ii, 4])
            Kstr = '  K = {0:0.5f}'.format(K) + r'$\pm$' + '{0:0.5f}'.format(fits[ii, 6])
            plt.title(r'$\omega = $' + str(eigval[ii]) + Astr + Kstr)
            if np.real(eigval[ii]) < 1.1:
                plt.show()
            elif 2.5 > np.real(eigval[ii]) > 2.1:
                plt.pause(1)
            else:
                plt.pause(0.001)

    # Inspect localization parameter K (exponent in exp(K * r)
    if check:
        plt.plot(np.real(eigval), fits[:, 2])
        plt.show()
    return fits


def fit_eigvect_to_exponential_periodic(xy, eigval, eigvect, LL, locutoffd=None, hicutoffd=None, check=False):
    """Fit the excitations of a gyro lattice to exponential decay (for detecting and measuring localized states)

    Parameters
    ----------
    xy : NP x 2 float array
        positions of particles in 2D
    eigval : 2*NP x 1 complex array
        eigenvalues of the system
    eigvect : 2*NP x 2*NP complex array
        eigenvectors of the system
    LL : tuple of floats (2 x 1 float array is allowed)
        spatial extent of the periodic system in 2d
    cutoffd : float
        cutoff distance for fitting exponential
    check : bool
        show intermediate steps

    Returns
    -------
    fits : NP x 7 float array (ie, int(len(eigval)*0.5) x 7 float array)
        fit details: x_center, y_center, A, K, uncertainty_A, covariance_AK, uncertainty_K
    """
    # Contract x and y component of each evect so that len(eigvect[ii]) = NP
    magevec = np.zeros((len(eigval), int(len(eigval) * 0.5)), dtype=float)
    jj = 0
    for row in eigvect:
        magevec[jj] = np.array([np.sqrt(np.abs(row[ii])**2 + np.abs(row[2*ii + 1])**2)
                                for ii in np.arange(0, int(len(eigval) * 0.5))])
        jj += 1

    # find COM for each eigvect
    fits = np.zeros((len(eigval), 7), dtype=float)
    for ii in range(len(eigval)):
        if ii % 100 == 1:
            print 'glfns: eval #', str(ii), '/', len(eigval)
        com = le.com_periodic(xy, LL, magevec[ii])
        fits[ii, 0:2] = com

        # Get minimum distance for each particle, taking care to get minimum across periodic BCs
        dist = le.distance_periodic(xy, com, LL)

        # Check it
        inds = np.argsort(dist)
        if hicutoffd is not None:
            inds = inds[dist[inds] < hicutoffd]
        if locutoffd is not None:
            inds = inds[dist[inds] > locutoffd]
        mags = magevec[ii][inds]
        dists = dist[inds]

        # Check if exponential is possible
        # kstest_exponential(magevec[ii], dist, bins=int(np.max(dist)))
        # possibly_exp = possibly_exponential(dists, mags)

        # if possibly_exp:
        # Fit to an exponential y=A*exp(K*t) using linear method.
        # METHOD 1
        # bin_means, bin_edges, bin_number = scistat.binned_statistic(dists, mags, bins=int(cutoffd))
        # A, K, cov = fit_exp_linear(binc, bin_means)
        A, K, cov = fit_exp_linear(dists, mags)
        fits[ii, 2] = A
        fits[ii, 3] = K
        fits[ii, 4] = cov[0, 0]
        fits[ii, 5] = cov[0, 1]
        fits[ii, 6] = cov[1, 1]

        # check the result
        if check:
            plt.plot(dists, mags, 'b.-')
            Astr = '  A = {0:0.5f}'.format(A) + r'$\pm$' + '{0:0.5f}'.format(np.sqrt(fits[ii, 4]))
            Kstr = '  K = {0:0.5f}'.format(K) + r'$\pm$' + '{0:0.5f}'.format(np.sqrt(fits[ii, 6]))
            plt.title(r'$\omega = $' + str(eigval[ii]) + Astr + Kstr)
            if np.real(eigval[ii]) < 1.1:
                plt.show()
            elif 2.5 > np.real(eigval[ii]) > 2.1:
                plt.pause(1)
            else:
                plt.pause(0.001)

    # Inspect localization parameter K (exponent in exp(K * r)
    if check:
        plt.plot(np.real(eigval), fits[:, 2])
        plt.show()
    return fits


# def fit_eigvect_to_exponential_periodicstrip(xy, eigval, eigvect, LL, cutoffd=None, check=False):
#     """Fit the excitations of a gyro lattice to exponential decay (for detecting and measuring localized states)
#
#     Parameters
#     ----------
#     xy : NP x 2 float array
#         positions of particles in 2D
#     eigval : 2*NP x 1 complex array
#         eigenvalues of the system
#     eigvect : 2*NP x 2*NP complex array
#         eigenvectors of the system
#     LL : float
#         width of the periodic system in 2d
#     cutoffd : float
#         cutoff distance for fitting exponential
#     check : bool
#         show intermediate steps
#
#     Returns
#     -------
#     fits : NP x 7 float array (ie, int(len(eigval)*0.5) x 7 float array)
#         fit details: x_center, y_center, A, K, uncertainty_A, covariance_AK, uncertainty_K
#     """
#     # convert eigvect to magnitude of eigenvectors, consider only half of eigvals
#     eigval = eigval[int(len(eigval) * 0.5):]
#     eigvect = eigvect[int(len(eigval) * 0.5):]
#
#     # Contract x and y component of each evect so that len(eigvect[ii]) = NP
#     magevec = np.zeros((len(eigvect), len(eigvect)), dtype=float)
#     jj = 0
#     for row in eigvect:
#         magevec[jj] = np.array([np.sqrt(np.abs(row[2*ii])**2 + np.abs(row[2*ii + 1])**2)
#                                 for ii in np.arange(0, int(len(eigvect) * 0.5)))])
#         jj += 1
#
#     # find COM for each eigvect
#     fits = np.zeros((len(eigval), 7), dtype=float)
#     for ii in range(len(eigval)):
#         if ii % 100 == 1:
#             print 'glfns: eval #', str(ii), '/', len(eigval)
#         com = le.com_periodicstrip(xy, LL, magevec[ii])
#         fits[ii, 0:2] = com
#
#         # Get minimum distance for each particle, taking care to get minimum across periodic BCs
#         dist = le.distance_periodicstrip(xy, com, LL)
#
#         # Check it
#         inds = np.argsort(dist)
#         if cutoffd is not None:
#             inds = inds[dist[inds] < cutoffd]
#         mags = magevec[ii][inds]
#         dists = dist[inds]
#
#         # Check if exponential is possible
#         # kstest_exponential(magevec[ii], dist, bins=int(np.max(dist)))
#         # possibly_exp = possibly_exponential(dists, mags)
#
#         # if possibly_exp:
#         # Fit to an exponential y=A*exp(K*t) using linear method.
#         # METHOD 1
#         # bin_means, bin_edges, bin_number = scistat.binned_statistic(dists, mags, bins=int(cutoffd))
#         # A, K, cov = fit_exp_linear(binc, bin_means)
#         A, K, cov = fit_exp_linear(dists, mags)
#         fits[ii, 2] = A
#         fits[ii, 3] = K
#         fits[ii, 4] = cov[0, 0]
#         fits[ii, 5] = cov[0, 1]
#         fits[ii, 6] = cov[1, 1]
#
#         # check the result
#         if check:
#             plt.plot(dists, mags, 'b.-')
#             Astr = '  A = {0:0.5f}'.format(A) + r'$\pm$' + '{0:0.5f}'.format(np.sqrt(fits[ii, 4]))
#             Kstr = '  K = {0:0.5f}'.format(K) + r'$\pm$' + '{0:0.5f}'.format(np.sqrt(fits[ii, 6]))
#             plt.title(r'$\omega = $' + str(eigval[ii]) + Astr + Kstr)
#             if np.real(eigval[ii]) < 1.1:
#                 plt.show()
#             elif 2.5 > np.real(eigval[ii]) > 2.1:
#                 plt.pause(1)
#             else:
#                 plt.pause(0.001)
#
#     # Inspect localization parameter K (exponent in exp(K * r)
#     if check:
#         plt.plot(np.real(eigval), fits[:, 2])
#         plt.show()
#     return fits


def calc_magevecs(eigvect):
    """Compute the magnitude all eigenvectors, by norming their x and y components in quad"""
    # Contract x and y component of each evect so that len(eigvect[ii]) = NP
    magevecs = np.zeros((int(len(evect) * 0.5), int(len(evect) * 0.5)), dtype=float)
    jj = 0
    for row in evect:
        magevecs[jj] = np.array([np.sqrt(np.abs(row[2 * ii]) ** 2 + np.abs(row[2 * ii + 1]) ** 2)
                                for ii in np.arange(0, int(len(eigvect) * 0.5))])
        jj += 1

    return magevecs


def fit_eigvect_to_exponential_1dperiodic(xy, eigval, eigvect, LL, locutoffd=None, hicutoffd=None, check=False):
    """Fit the excitations of a gyro lattice to exponential decay, using
    Delta = lim_x->infty 1/x ln(psi(x))

    Parameters
    ----------
    xy : NP x 2 float array
        positions of particles in 2D
    eigval : 2*NP x 1 complex array
        eigenvalues of the system
    eigvect : 2*NP x 2*NP complex array
        eigenvectors of the system
    LL : float
        width of the periodic system in 2d
    locutoffd : float
        low-end cutoff distance for fitting exponential
    hicutoffd : float
        hi-end cutoff distance for fitting exponential
    check : bool
        show intermediate steps

    Returns
    -------
    fits : NP x 7 float array (ie, int(len(eigval)*0.5) x 7 float array)
        fit details: x_center, y_center, A, K, uncertainty_A, covariance_AK, uncertainty_K
    """
    # convert eigvect to magnitude of eigenvectors
    # Contract x and y component of each evect so that len(eigvect[ii]) = NP
    magevec = calc_magevecs(eigvect)

    # find COM for each eigvect
    fits = np.zeros((len(eigval), 7), dtype=float)
    for ii in range(len(eigval)):
        if ii % 100 == 1:
            print 'glfns: eval #', str(ii), '/', len(eigval)

        # le.plot_real_matrix(magevec, show=True)
        # print 'ii = ', ii
        # print 'np.shape(magevec[ii]) = ', np.shape(magevec[ii])
        # print 'np.min(magevec[ii]) = ', np.min(magevec[ii])
        # print 'np.max(magevec[ii]) = ', np.max(magevec[ii])
        com = le.com_periodicstrip(xy, LL, masses=magevec[ii])
        fits[ii, 0:2] = com
        # print 'com =', com

        # Get minimum distance for each particle, taking care to get minimum across periodic BCs
        dist = le.distancex_periodicstrip(xy, com, LL)

        # Check it
        inds = np.argsort(dist)
        if hicutoffd is not None:
            inds = inds[dist[inds] < hicutoffd]
        if locutoffd is not None:
            inds = inds[dist[inds] < locutoffd]
        mags = magevec[ii][inds]
        dists = dist[inds]

        # Check if exponential is possible
        # kstest_exponential(magevec[ii], dist, bins=int(np.max(dist)))
        # possibly_exp = possibly_exponential(dists, mags)

        # if possibly_exp:
        # Fit to an exponential y=A*exp(K*t) using linear method.
        # METHOD 1
        # bin_means, bin_edges, bin_number = scistat.binned_statistic(dists, mags, bins=int(cutoffd))
        # A, K, cov = fit_exp_linear(binc, bin_means)
        A, K, cov = fit_exp_linear(dists, mags)
        fits[ii, 2] = A
        fits[ii, 3] = K
        fits[ii, 4] = cov[0, 0]
        fits[ii, 5] = cov[0, 1]
        fits[ii, 6] = cov[1, 1]

        # check the result
        if check:
            if np.abs(K) > 0.1:
                plt.clf()
                plt.plot(dists, mags, 'b.-')
                plt.plot(dists, A * np.exp(K*dists), 'r-')
                Astr = '  A = {0:0.5f}'.format(A) + r'$\pm$' + '{0:0.5f}'.format(np.sqrt(fits[ii, 4]))
                Kstr = '  K = {0:0.5f}'.format(K) + r'$\pm$' + '{0:0.5f}'.format(np.sqrt(fits[ii, 6]))
                plt.title(r'$\omega = $' + str(eigval[ii]) + Astr + Kstr)

                plt.show()
                com = le.com_periodicstrip(xy, LL, magevec[ii], check=True)
            # if np.real(eigval[ii]) > 3.7:
            #     # plt.show()
            #     pass
            # else:
            #     plt.pause(0.0001)
            #     pass

    # Inspect localization parameter K (exponent in exp(K * r)
    if check:
        plt.plot(np.real(eigval), fits[:, 2])
        plt.show()

    return fits


def fit_edgedecay_periodicstrip(xy, eigval, eigvect, cutoffd=None, check=False):
    """Fit the excitations of a gyro lattice to exponential decay (for detecting and measuring localized states)

    Parameters
    ----------
    xy : NP x 2 float array
        positions of particles in 2D
    eigval : 2*NP x 1 complex array
        eigenvalues of the system
    eigvect : 2*NP x 2*NP complex array
        eigenvectors of the system
    LL : tuple of floats (2 x 1 float array is allowed)
        spatial extent of the periodic system in 2d
    cutoffd : float
        cutoff distance for fitting exponential
    check : bool
        show intermediate steps

    Returns
    -------
    fits : NP x 7 float array (ie, int(len(eigval)*0.5) x 7 float array)
        fit details: x_center, y_center, A, K, uncertainty_A, covariance_AK, uncertainty_K
    """
    # convert eigvect to magnitude of eigenvectors
    # Contract x and y component of each evect so that len(eigvect[ii]) = NP
    magevec = np.zeros((len(eigval), len(eigval)), dtype=float)
    jj = 0
    for row in eigvect:
        magevec[jj] = np.array([np.sqrt(np.abs(row[2*ii])**2 + np.abs(row[2*ii + 1])**2)
                                for ii in np.arange(0, len(eigval))])
        jj += 1

    # assume amplitude largest against the top or bottom edge
    fits = np.zeros((len(eigval), 7), dtype=float)
    for ii in range(len(eigval)):
        if ii % 100 == 1:
            print 'glfns: eval #', str(ii), '/', len(eigval)

        # first fit to bottom, then to top, keep better fit
        # Get distance from edge for each particle
        dist = xy[:, 1] - np.min(xy[:, 1])
        # Sort the results
        inds = np.argsort(dist)
        if cutoffd is not None:
            inds = inds[dist[inds] < cutoffd]
        mags = magevec[ii][inds]
        dists = dist[inds]
        Ab, Kb, covb = fit_exp_linear(dists, mags)

        # fit to top
        dist = np.max(xy[:, 1]) - xy[:, 1]
        inds = np.argsort(dist)
        if cutoffd is not None:
            inds = inds[dist[inds] < cutoffd]
        mags = magevec[ii][inds]
        dists = dist[inds]
        At, Kt, covt = fit_exp_linear(dists, mags)

        # Keep whichever has smaller K
        if Kt < Kb:
            fits[ii, 0:2] = np.array([0, np.min(xy[:, 1])])
            A = At
            K = Kt
            cov = covt
        else:
            fits[ii, 0:2] = np.array([0, np.max(xy[:, 1])])
            A = Ab
            K = Kb
            cov = covb

        fits[ii, 2] = A
        fits[ii, 3] = K
        fits[ii, 4] = cov[0, 0]
        fits[ii, 5] = cov[0, 1]
        fits[ii, 6] = cov[1, 1]

        # check the result
        if check:
            plt.plot(dists, mags, 'b.-')
            Astr = '  A = {0:0.5f}'.format(A) + r'$\pm$' + '{0:0.5f}'.format(np.sqrt(fits[ii, 4]))
            Kstr = '  K = {0:0.5f}'.format(K) + r'$\pm$' + '{0:0.5f}'.format(np.sqrt(fits[ii, 6]))
            plt.title(r'$\omega = $' + str(eigval[ii]) + Astr + Kstr)
            if np.real(eigval[ii]) < 1.1:
                plt.show()
            elif 2.5 > np.real(eigval[ii]) > 2.1:
                plt.pause(1)
            else:
                plt.pause(0.001)

    # Inspect localization parameter K (exponent in exp(K * r)
    if check:
        plt.plot(np.real(eigval), fits[:, 2])
        plt.show()


def possibly_exponential(xx, yy, bins=10):
    """
    """
    bin_means, bin_edges, bin_number = scistat.binned_statistic(xx, yy, bins=bins)
    # print 'bin_means = ', bin_means
    # print 'bin_edges = ', bin_edges
    # plt.plot(bin_edges[1:], np.log(bin_means), 'b.-')
    # plt.title('Possy exp = ' + str(bin_means[0] > np.mean(bin_means[1:])))
    # plt.pause(0.1)
    # plt.clf()
    return bin_means[0] > np.mean(bin_means[1:]) and bin_means[1] > np.mean(bin_means[2:])


def fit_exponential(t, y):
    """Fit data to y=A*exp(K*t) using linear method.

    """
    # Fitting to y = A * exp(K * t)
    # Linear Fit (Note that we have to provide the y-offset ("C") value!!
    A, K, cov = fit_exp_linear(t, y)
    fit_y = exponential_func(t, A, K, 0.)
    return A, K, cov, fit_y


def exponential_func(t, A, K, C):
    return A * np.exp(K * t) + C


def fit_exp_linear(t, y, C=0):
    """Fit data y(t) to y=A*exp(K*t) using a linear method by fitting y = log(A * exp(K * t)) = K * t + log(A).
    """
    K_Alog, cov = np.polyfit(t, np.log(y - C), 1, full=False, cov=True)
    K = K_Alog[0]
    A = np.exp(K_Alog[1])
    return A, K, cov


###################################
# Kolmogorov-Smirnoff test notes
###################################
def expcdf(t, beta=1.0):
    """"""
    return 1.0 - np.exp(-t/beta)


def kstest_exponential(mags, dist, bins=-1):
    """Perform Kolmogorov-Smirnoff test with data that may approximate an exponential curve.
    See http://stats.stackexchange.com/questions/110272/a-naive-question-about-the-kolmogorov-smirnov-test
    """
    if bins == -1:
        bins = int(np.max(dist))
    bin_means, bin_edges, bin_number = scistat.binned_statistic(mags, dist, bins=bins)
    print 'bin_edges = ', bin_edges
    xx = np.cumsum(bin_means)
    xx = (xx - xx.min()) / xx.ptp()
    xx = []
    jj = 0
    for be in bin_means:
        for ii in np.arange(int(100 * bin_means[jj])):
            xx.append(bin_edges[jj])
        jj += 1
    print 'xx = ', xx
    plt.hist(xx)
    plt.show()
    return scistat.kstest(xx, expcdf)  # 'expon')


# Example using fake data
# import numpy as np
# import scipy.stats as scistat
# import matplotlib.pyplot as plt
# dist = np.arange(100)/100.
# mags = np.exp(-dist)
# kstest_exponential(mags, dist, bins=12)
