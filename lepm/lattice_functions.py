import numpy as np
import lepm.lattice_elasticity as le
import lepm.plotting.colormaps as lecmaps
from matplotlib.collections import PatchCollection
import numpy.linalg as la
import matplotlib.pyplot as plt
import sys
import lepm.data_handling as dh
import lepm.line_segments as linsegs
import glob
import os
import lepm.brillouin_zone_functions as bzfns

"""Auxiliary functions for lattice_class.py methods"""


def complete_lp(lp):
    """Fill in missing (but necessary) key-value pairs in the lattice parameters dictionary

    Parameters
    ----------
    lp : dict

    Returns
    -------
    lp : dict
    """
    if 'periodic_strip' not in lp:
        lp['periodic_strip'] = False
    return lp


def param2meshfn_name(param):
    """Convert a parameter (string) name into its possibly-abbreviated form that appears as part of a meshfn (the
    string specifier pathname of a lattice).

    Parameters
    ----------
    param : str
        A key for a Lattice instance's lp (lattice parameter dictionary)

    Returns
    -------
    mfestr : str
    """
    if param in ['delta_lattice', 'delta']:
        mfestr = 'delta'
    else:
        raise RuntimeError('Have not yet supported this parameter in string conversion -- add line for it here.')

    return mfestr


def load(lat, meshfn='auto', load_polygons=False, load_LVUC=False, load_gxy=False, check=False):
    """Load a saved lattice into the lattice instance. If meshfn is specified, loads that lattice.
        Otherwise, attempts to load lattice based on parameter lat.lp['meshfn']. If that is also unavailable,
        loads from lp[rootdir]/networks/lat.LatticeTop/lat.lp[lattice_exten]_NH_x_NV.
    """
    print '\n\n\n\nLoading network: meshfn == ', meshfn
    if meshfn == 'auto':
        fn = lat.automeshfn()
    else:
        fnglob = sorted(glob.glob(meshfn))
        print 'globbing meshfn: fnglob = ', fnglob
        is_a_dir = np.where(np.array([os.path.isdir(ii) for ii in fnglob]))[0]
        print 'meshfn = ', meshfn
        fn = fnglob[is_a_dir[0]]
        # print 'is_a_dir = ', is_a_dir
        # print 'np.size(is_a_dir) = ', np.size(is_a_dir)
        # print 'fn = ', fn
        if np.size(is_a_dir) > 1:
            print 'Found multiple lattices matching meshfn in lattice.load(). Using the first matching lattice.'
            fn = fn[0]

        lat.lp['meshfn'] = fn

    if fn[-1] == '/':
        fn = fn[:-1]
    lat.lp = le.load_params(fn + '/', 'lattice_params', ignore=physics_to_ignore())
    if not glob.glob(lat.lp['meshfn']):
        print "\n\nlattice_class: Warning: could not find lp[meshfn] = \n" + \
              lat.lp['meshfn'] + '\nso instead replacing lp[meshfn] with --->\n' + fn + '\n\n'
        lat.lp['meshfn'] = fn

    # Fix up the lattice parameters dictionary by adding any necessary key-vals that aren't already in there
    lat.lp = complete_lp(lat.lp)

    lat.xy = np.loadtxt(fn + '_xy.txt', delimiter=',')
    lat.NL = np.loadtxt(fn + '_NL.txt', delimiter=',', dtype=int)
    lat.KL = np.loadtxt(fn + '_KL.txt', delimiter=',', dtype=int)
    try:
        lat.BL = np.loadtxt(fn + '_BL.txt', delimiter=',', dtype=int)
    except IOError:
        lat.BL = le.NL2BL(lat.NL, lat.KL)

    # Make sure that BL is 2D, even if only one row
    if len(np.shape(lat.BL)) < 2:
        lat.BL = np.array([lat.BL])

    name = fn.split('/')[-1]
    try:
        print 'Lattice: loading ' + name + '_PVxydict/Pvx/Pvy.txt ...'
        lat.PVxydict = le.load_evaled_dict(fn + '/', filename=name + '_PVxydict.txt')
        lat.PVx = np.loadtxt(fn + '/' + name + '_PVx.txt', delimiter=',', dtype=float)
        lat.PVy = np.loadtxt(fn + '/' + name + '_PVy.txt', delimiter=',', dtype=float)
        if (lat.PVx == lat.PVy).all():
            print 'PVx and PVy are identical (this was a bug in early versions of the code), need to trash one: ' \
                  'overwriting then exiting.'
            lat.PVx, lat.PVy = le.PVxydict2PVxPVy(lat.PVxydict, lat.NL)
            header = 'PVx: ijth element of PVx are the x-components of the vector taking NL[i,j] to ' + \
                     'its image as seen by particle i'
            np.savetxt(fn + '/' + name + '_PVx.txt', lat.PVx, delimiter=',', header=header)
            header = 'PVy: ijth element of PVy are the y-components of the vector taking NL[i,j] to ' + \
                     'its image as seen by particle i'
            np.savetxt(fn + '/' + name + '_PVy.txt', lat.PVy, delimiter=',', header=header)
            print 'lattice_class: exiting here'
            sys.exit()

        # load periodic vectors if they exists
        try:
            lat.load_PV(meshfn=fn)
        except IOError:
            print 'Lattice: no PV file exists, saving it...'
            lat.PV = le.PVxydict2PV(lat.PVxydict)
            lat.save_PV()
    except:
        print 'lattice_class: Could not load PVx and PVy, loading PVxydict to calculate them...'
        # print ' --> actually, just exiting since this should not happen anymore!'
        # sys.exit()
        try:
            lat.PVxydict = le.load_evaled_dict(fn + '/', filename=name + '_PVxydict.txt')
            lat.PVx, lat.PVy = le.PVxydict2PVxPVy(lat.PVxydict, lat.NL)
            header = 'PVx: ijth element of PVx are the x-components of the vector taking NL[i,j] to ' + \
                     'its image as seen by particle i'
            np.savetxt(fn + '/' + name + '_PVx.txt', lat.PVx, delimiter=',', header=header)
            header = 'PVy: ijth element of PVy are the y-components of the vector taking NL[i,j] to ' + \
                     'its image as seen by particle i'
            np.savetxt(fn + '/' + name + '_PVy.txt', lat.PVy, delimiter=',', header=header)
            try:
                lat.load_PV(meshfn=fn)
            except IOError:
                lat.PV = le.PVxydict2PV(lat.PVxydict)

        except IOError:
            print 'No periodic vectors stored for this lattice, assuming not periodic.'

    # Load the boundary of the sample, since that is cheap (sqrt(N) at most)
    single_bndy = glob.glob(fn + '/' + name + '_boundary.txt')
    double_bndy = glob.glob(fn + '/' + name + '_boundary0.txt')
    if single_bndy:
        boundary = np.loadtxt(fn + '/' + name + '_boundary.txt', delimiter=',', dtype=int)
        lat.boundary = boundary
    elif double_bndy:
        boundary0 = np.loadtxt(fn + '/' + name + '_boundary0.txt', delimiter=',', dtype=int)
        boundary1 = np.loadtxt(fn + '/' + name + '_boundary1.txt', delimiter=',', dtype=int)
        boundary = (boundary0, boundary1)
        lat.boundary = boundary
    else:
        if not lat.lp['periodicBC'] or lat.lp['periodic_strip']:
            print 'No boundary file found for (partially/fully) openBC network, saving the boundary to file...'
            bndry = lat.get_boundary(attribute=True, check=check)
            print 'bndry = ', bndry
            if isinstance(bndry, tuple):
                np.savetxt(fn + '/' + name + '_boundary0.txt', bndry[0], fmt='%d',
                           header="indices of one of the network's boundaries -- boundary zero")
                np.savetxt(fn + '/' + name + '_boundary1.txt', bndry[1], fmt='%d',
                           header="indices of one of the network's boundaries -- boundary one")
            else:
                np.savetxt(fn + '/' + name + '_boundary.txt', bndry, fmt='%d',
                           header='indices of the network boundary')
            # print image of the boundaries
            lat.plot_boundary(show=False, save=True, outdir=fn + '/', outname=name + '_boundary.png')

    if load_LVUC:
        lat.load_LUVC()

    if load_polygons:
        lat.load_polygons()

    if load_gxy:
        lat.load_gxy()


def plot_polygons_with_nsides(lat, nsides, ax, color=None, alpha=0.5):
    """For Lattice instance lat, plot all polygons with n sides on axis ax

    Parameters
    ----------
    lat : lepm.lattice_class.LatticeClass instance
        lattice class instance with attributes xy, BL, PVxydict, etc

    """
    polygons = lat.get_polygons()
    pnsides = []
    for poly in polygons:
        if len(poly) == nsides + 1:
            pnsides.append(poly)

    ppc = le.polygons2PPC(lat.xy, pnsides, BL=lat.BL, PVxydict=lat.PVxydict, check=False)
    p = PatchCollection(ppc, facecolors=color, edgecolors='none', alpha=alpha)
    ax.add_collection(p)


def physics_to_ignore():
    """Retun keys which we should ignore when loading a lattice_parameters.txt file, so that we don't impart physics to
    a generic, geometric network

    Returns
    -------
    ignore : list of strings
    """
    ignore = ['V0_pin_gauss', 'V0_spring_gauss', 'pin', 'Omg', 'Omk', 'OmK']
    return ignore


def calc_PVxyij(lat):
    """

    Parameters
    ----------
    lat : lepm.lattice_class.LatticeClass instance
        lattice class instance

    Returns
    -------
    PVxij : NP x NP float array (optional, for periodic lattices)
        ijth element of PVxij is the x-component of the vector taking particle j (at xy[j]) to its image as seen by
        particle i (at xy[i])
    PVyij : NP x NP float array (optional, for periodic lattices)
        ijth element of PVyij is the x-component of the vector taking particle j (at xy[j]) to its image as seen by
        particle i (at xy[i])
    """
    PV = lat.get_PV()
    return dh.dist_pts_periodic(lat.xy, lat.xy, PV, dim=0), dh.dist_pts_periodic(lat.xy, lat.xy, PV, dim=1)


def calc_PV(lat):
    """Compute the periodic vectors for a periodic boundary condition system. See also le.PVxydict2PV() for a more
    robust method.

    Parameters
    ----------
    lat : lepm.lattice_class.LatticeClass instance
        lattice class instance

    Returns
    -------
    PV : 2 x 2 float array
        periodic lattice vectors, with x-dominant vector first, y-dominant vector second.
    """
    bbox = lat.lp['BBox']
    if lat.PVxydict is not None:
        PV = le.PVxydict2PV(lat.PVxydict)  # , check=True)
    else:
        # otherwise let's use the bounding box and assume it is square
        right = np.where(bbox[:, 0] == np.max(bbox[:, 0]))[0]
        left = np.where(bbox[:, 0] == np.min(bbox[:, 0]))[0]
        top = np.where(bbox[:, 1] == np.max(bbox[:, 1]))[0]
        bot = np.where(bbox[:, 1] == np.min(bbox[:, 1]))[0]
        print 'bbox = ', bbox
        print 'right = ', right
        print 'left = ', left
        print 'top = ', top
        print 'bot = ', bot
        if (np.array([len(right), len(left), len(top), len(bot)]) == 2).all():
            # The periodic sample is a square, not sheared in any way
            PV = np.array([[np.max(bbox[:, 0]) - np.min(bbox[:, 0]), 0.], [0., np.max(bbox[:, 1]) - np.min(bbox[:, 1])]])
        else:
            raise RuntimeError('Have not coded for rhombic PVs yet --> do that here')

    return PV


def calc_bz(lat):
    """Get vertices of the brillouin zone for this lattice

    Parameters
    ----------
    lat : Lattice class

    Returns
    -------
    bz : n x 2 float array
        brillouin zone vertices for the current network (periodic)
    """
    if lat.lp['LV'] is not None:
        bzx, bzy = bzfns.bz_vertices(lat.lp['LV'][0], lat.lp['LV'][1])
    else:
        bzx, bzy, rr = bzfns.bz_based_on_ps(lat.xy)

    bz = np.dstack((bzx, bzy))[0]
    return bz


def calc_nn_of_sites(lat, site_inds):
    """Compute all nearest neighbors of the sites indexed by site_inds

    Parameters
    ----------
    lat : Lattice class instance
        the lattice for which to compute the nearest neighbors of the sites indexed by site_inds
    site_inds : n x 1 int array
        the sites to which we meausre the minimum distance of all particles lat.xy

    Returns
    -------
    nnsites : m x 1 int array
        the indices of lat.xy which are nearest neighbors of lat.xy[site_inds]
    """
    nnsites = []
    for kk in site_inds:
        nlrow = lat.NL[kk, np.where(np.abs(lat.KL[kk]))[0]]
        for site in nlrow:
            if site not in nnsites:
                nnsites.append(site)
    return np.array(nnsites)


def reciprocal_latvecs(latvecs):
    """Compute the reciprocal lattice vectors from the primitive lattice vectors

    Parameters
    ----------
    latvecs : float array or list of length d with 1 x d float arrays as elements, where d is the dimension of space
        the lattice vectors

    Returns
    -------
    """
    # First check the dimensionality
    if len(latvecs[0]) == 2:
        # define 90-degree rotation matrix
        rotm = np.array([[0, -1], [1, 0]])
        a1 = np.array(latvecs[0])
        a2 = np.array(latvecs[1])
        b1 = - 2. * np.pi * np.dot(rotm, a2) / (- np.dot(a1, np.dot(rotm, a2)))
        b2 = - 2. * np.pi * np.dot(rotm, a1) / (- np.dot(a2, np.dot(rotm, a1)))
        rlv = np.vstack((b1, b2))

        # a1 = np.array(latvecs[0])
        # a2 = np.array(latvecs[1])
        # a3 = np.array([0, 0, 1])
        # print 'a1 = ', a1
        # print 'a2 = ', a2
        # print 'a3 = ', a3
        # b1 = 2 * np.pi * np.cross(a2, a3) / (np.dot(a1, np.cross(a2, a3)))
        # b2 = 2 * np.pi * np.cross(a3, a1) / (np.dot(a1, np.cross(a2, a3)))
        # rlv = np.vstack((b1, b2))
    else:
        raise RuntimeError('lepm.lattice_functions.reciprocal_latvecs() currently only supports 2D lattices')

    return rlv


def bz_from_latvecs(latvecs):
    """

    Parameters
    ----------
    latvecs

    Returns
    -------
    bz : #vertices x 2 float array
        The vertices of the brillouin zone, returned in counterclockwise order
    """
    rlv = reciprocal_latvecs(latvecs)
    print 'rlv = ', rlv
    # Create a network of lines bisecting the paths from origin to each other reciprocal lattice point
    # First create local reciprocal lattice points
    vals = np.arange(-3, 3)
    rp = []
    for ii in vals:
        for jj in vals:
            rp.append(ii * rlv[0] + jj * rlv[1])

    # Delete the zero vector from rp
    rp = np.array(rp)
    print 'rp = ', rp
    rp = dh.setdiff2d(rp, np.array([[0., 0.]]))

    # create bisecting lines
    perps = linsegs.perp_vects(rp)
    midpts = 0.5 * rp
    lines = np.hstack((midpts, midpts + perps))
    print 'lines = ', lines
    # The function below executes:
    # Get the intersections of the lines, call them the vertices
    # Place 'bonds' between vertices
    vtcs, bl = linsegs.network_from_intersections(lines)

    # Find polygons of this network
    polygons = le.extract_polygons_lattice(vtcs, bl)

    # The polygon containing the origin is the first BZ
    inds = dh.polygons_enclosing_pt(np.array([0, 0]), polygons)
    # There should just be a single index in the output inds
    if len(inds) > 1:
        print 'polygons = ', polygons
        print 'inds = ', inds
        print 'vtcs = ', vtcs
        print 'bl = ', bl
        raise RuntimeError('Found more than one polygon encloses origin! Check the network that is built, '
                           'printed above.')
    bz = polygons[inds[0]]
    # Roll output polygon (BZ) so that first point is close to positive side of x axis (sorted counterclockwise)
    print 'bz = ', bz
    print 'np.argmin(np.atan2(bz[:, 1], bz[:, 0]) % (2. * np.pi)) = ', np.argmin(np.atan2(bz[:, 1], bz[:, 0]) % (2. * np.pi))
    bz = np.roll(bz, len(bz) - np.argmin(np.atan2(bz[:, 1], bz[:, 0]) % (2. * np.pi)))
    print 'rolled bz = ', bz

    return bz


if __name__ == '__main__':
    lv = np.array([[1., 0], [0., 1.]])
    bz = bz_from_latvecs(lv)
    plt.plot(np.hstack((bz[:, 0], np.array([bz[0, 0]]))), np.hstack((bz[:, 1], np.array([bz[0, 1]]))))