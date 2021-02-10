from chern_functions_gen import *
from numpy import *
import os
from matplotlib.path import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.integrate import dblquad
from matplotlib import cm
import time
import itertools
import os
import os.path
from scipy.interpolate import griddata
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage
import bz_funcs as bz
import tight_binding_functions as tbf
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import pylab


def itertools_iterable_flatten(iter_iter):
    return list(itertools.chain.from_iterable(iter_iter))


def alpha_lattice(t1234, ons, base_dir='/Users/lisa/Dropbox/Research/New_Chern_general/data/'):
    """This function creates an `alpha' lattice. (This is a lattice geometry).
    
    Parameters
    ----------
    t1234 : array of length 4
        The 4 values for Omega_k of the different springs.  

    ons: array of length 4
        The on-site Omega_g value for the A, B, C, & D sites respectively
        
    Returns
    ----------
    mM : function of two variables
        Function which calculates the interaction matrix for this lattice

    vertex_points : array of dimension 6x2
        points which trace out the boundary of the Brillouin zone

    od : string
        output directory for data based on lattice type
        
    fn : string
        file name

    [4, 4] : array
            particular to this lattice.  Number of different bond strengths and on-site energies.
    """
    t1 = t1234[0]
    t2 = t1234[1]
    t3 = t1234[2]
    t4 = t1234[3]
    pin = 1
    th = 60 * pi / 180.
    nA = array([[0, 1], [0, -1], [-1 * cos(th / 2), 1 * sin(th / 2)], [-1 * cos(th / 2), -1 * sin(th / 2)]])  # B B C D
    nB = array([[0, 1], [0, -1], [1 * cos(th / 2), -1 * sin(th / 2)], [1 * cos(th / 2), 1 * sin(th / 2)]])  # A A C D
    nC = array([[1 * cos(th / 2), -1 * sin(th / 2)], [-1 * cos(th / 2), 1 * sin(th / 2)]])  # A B
    nD = array([[1 * cos(th / 2), 1 * sin(th / 2)], [-1 * cos(th / 2), -1 * sin(th / 2)]])  # A B

    g = lambda arr: [arctan2(arr[i, 1], arr[i, 0]) for i in range(len(arr))]
    f = lambda arr: [1 for i in range(len(arr))]

    angs = [g(nA), g(nB), g(nC), g(nD)
            ]
    bls = [f(nA), f(nB), f(nC), f(nD)]

    tvals = [[t2, t1, t4, t3],
             [t1, t2, t4, t3],
             [t4, t4],
             [t3, t3]]

    num_neighbors = [[0, 2, 1, 1],
                     [2, 0, 1, 1],
                     [1, 1, 0, 0],
                     [1, 1, 0, -0]
                     ]

    a1, a2 = bz.find_lattice_vecs_alpha_ns()
    b1, b2 = bz.find_bz_lattice_vecs(a1, a2)
    vtx, vty = bz.find_bz_zone_alpha(b1, b2)
    vertex_points = array([vtx, vty]).T

    mM = calc_matrix(angs, num_neighbors, bls, tvals, ons)

    s1 = '%0.2f_' % t1234[0]
    s2 = '%0.2f_' % t1234[1]
    s3 = '%0.2f_' % t1234[2]
    s4 = '%0.2f_' % t1234[3]

    s5 = '%0.2f_' % ons[0]
    s6 = '%0.2f_' % ons[1]
    s7 = '%0.2f_' % ons[2]
    s8 = '%0.2f_' % ons[3]

    old_fn = 'data_dict' + s1 + s2 + s3 + s4 + '__' + s5 + s6 + s7 + s8

    fn = 'alpha_lattice_'

    fn += '_%0.2f_' % t1234[0]
    fn += '%0.2f_' % t1234[1]
    fn += '%0.2f_' % t1234[2]
    fn += '%0.2f__' % t1234[3]

    fn += '%0.2f_' % ons[0]
    fn += '%0.2f_' % ons[1]
    fn += '%0.2f_' % ons[2]
    fn += '%0.2f_' % ons[3]

    od = base_dir + '/alpha_lattice/'
    if not os.path.isdir(od):
        os.mkdir(od)
        os.mkdir(od + 'data/')
        os.mkdir(od + 'images/')
    return mM, vertex_points, od, [fn, old_fn], [4, 4]


def alpha_lattice_simp(t1234):
    """doesn't work yet"""
    pin = 1
    delta = 1
    ons = [1, 1, 1, 1]
    t1 = t1234[0]
    t2 = t1234[1]
    t3 = t1234[2]
    t4 = t1234[3]

    th = 60 * pi / 180.
    nA = array([[0, 1], [0, -1], [-2 * cos(th / 2), 2 * sin(th / 2)], [-2 * cos(th / 2), -2 * sin(th / 2)]])
    nB = array([[0, 1], [0, -1], [2 * cos(th / 2), 2 * sin(th / 2)], [2 * cos(th / 2), -2 * sin(th / 2)]])

    A = [0, 0]
    B = nA[1]

    bls = [[1, 1, 2, 2], [1, 1, 2, 2]]
    angs = [[arctan2(nA[0, 1], nA[0, 0]), arctan2(nA[1, 1], nA[1, 0]), arctan2(nA[2, 1], nA[2, 0]),
             arctan2(nA[3, 1], nA[3, 0])],
            [arctan2(nB[0, 1], nB[0, 0]), arctan2(nB[1, 1], nB[1, 0]), arctan2(nB[2, 1], nB[2, 0]),
             arctan2(nB[3, 1], nB[3, 0])]]  # B (L) A (R)
    num_neighbors = [[0, 4],
                     [4, 0]]

    ons = [1, 1]
    t1 = 1
    t2 = 1
    t3 = 0.5
    t4 = 0.5

    tvals = [[t2, t1, t4, t3],
             [t1, t2, t3, t4]]

    b1, b2 = bz.find_bz_lattice_vecs_alpha()
    v_pts, midpt_x, midpt_y, x, l, vtx, vty = bz.find_bz_zone_alpha(b1, b2)
    vertex_points = array([vtx, vty]).T

    mM = calc_matrix(angs, num_neighbors, bls, tvals, ons)

    return mM, vertex_points, './', 'alpha_lattice_simp'


def sq_lattice_4(t1234, pin, ons):
    """Initiates an anisotropic square lattice.
    
    Parameters
    ----------
    t1234 : array of length 4
        The 4 values for Omega_k of the different springs.  

    ons: array of length 4
        The on-site Omega_g value for the A, B, C, & D sites respectively
        
    Returns
    ----------
    mM : function of two variables
        Function which calculates the interaction matrix for this lattice

    vertex_points : array of dimension 6x2
        points which trace out the boundary of the Brillouin zone

    od : string
        output directory for data based on lattice type
        
    fn : string
        file name

    
    [4, 4] : array
            particular to this lattice.  Number of different bond strengths and on-site energies.
    """

    delta = 1

    t1 = t1234[0]
    t2 = t1234[1]
    t3 = t1234[2]
    t4 = t1234[3]

    th = 0 * pi / 180.
    nA = array([[sin(th), cos(th)], [-sin(th), -cos(th)], [1, 0], [-1, 0]])
    nB = array([[sin(th), cos(th)], [-sin(th), -cos(th)], [1, 0], [-1, 0]])
    nC = array([[1, 0], [-1, 0], [sin(th), cos(th)], [-sin(th), -cos(th)]])
    nD = array([[1, 0], [-1, 0], [sin(th), cos(th)], [-sin(th), -cos(th)]])

    g = lambda arr: [arctan2(arr[i, 1], arr[i, 0]) for i in range(len(arr))]
    f = lambda arr: [1 for i in range(len(arr))]

    angs = [g(nA), g(nB), g(nC), g(nD)
            ]
    bls = [f(nA), f(nB), f(nC), f(nD)]

    tvals = [[t3, t3, t2, t1],
             [t3, t3, t1, t2],
             [t2, t1, t4, t4],
             [t1, t2, t4, t4]]

    num_neighbors = [[0, 2, 0, 2],
                     [2, 0, 2, 0],
                     [0, 2, 0, 2],
                     [2, 0, 2, 0]
                     ]

    a1, a2 = bz.find_lattice_vecs_alpha_ns()
    b1, b2 = bz.find_bz_lattice_vecs(a1, a2)
    vtx, vty = bz.find_bz_zone_alpha(b1, b2)
    vertex_points = array([vtx, vty]).T

    mM = calc_matrix(angs, num_neighbors, bls, tvals, ons)

    return mM, vertex_points, od, fn


def honeycomb_sheared(t1234, d, p, ons, base_dir='/Users/lisa/Dropbox/Research/New_Chern_general/data/'):
    """Initiates a distorted and sheared honeycomb lattice.
    
    Parameters
    ----------
    t1234 : array of length 1
    
        The value for Omega_k 
        
    d : float
        The value of the angle delta in degrees
        
    p : float
        The value of the angle phi in degrees

    ons: array of length 2
        The on-site Omega_g value for the A and B sites respectively
        
    Returns
    ----------
    mM : function of two variables
        Function which calculates the interaction matrix for this lattice
    vertex_points : array of dimension 6x2
        points which trace out the boundary of the Brillouin zone
    od : string
        output directory for data based on lattice type
    fn : string
        file name
    [1, 2] : array
        particular to this lattice.  Number of different bond strengths and on-site energies respectively
    """
    t1 = t1234[0]

    delta = d * pi / 180
    phi = p * pi / 180.
    theta = 0.5 * (2 * pi - delta) - pi / 2

    nA = array([[cos(theta), sin(theta)], [-cos(theta), sin(theta)], [-sin(phi), -cos(phi)]])
    nB = array([[cos(theta), -sin(theta)], [-cos(theta), -sin(theta)], [sin(phi), cos(phi)]])

    A = [0, 0]
    B = nA[2]

    angs = [[arctan2(nA[0, 1], nA[0, 0]), arctan2(nA[1, 1], nA[1, 0]), arctan2(nA[2, 1], nA[2, 0])],
            [arctan2(nB[0, 1], nB[0, 0]), arctan2(nB[1, 1], nB[1, 0]), arctan2(nB[2, 1], nB[2, 0])]]  # B (L) A (R)

    bls = [[1, 1, 1], [1, 1, 1]]

    num_neighbors = [[0, 3],
                     [3, 0]]

    tvals = [[t1, t1, t1],
             [t1, t1, t1]]

    a1, a2 = bz.find_lattice_vecs(delta, phi)
    b1, b2 = bz.find_bz_lattice_vecs(a1, a2)
    vtx, vty = bz.find_bz_zone(b1, b2)
    vertex_points = array([vtx, vty]).T

    # Check
    # print 'in chern/lattice_functions.py...'
    # print 'a1, a2 = ', (a1, a2)
    # print 'b1, b2 = ', (b1, b2)
    # print 'vtx, vty = ', (vtx, vty)
    # a1, a2 = np.array([1.732050807568877193, 0., 0.]), np.array([ 0.866025403784438597, 1.5, 0.])
    # b1, b2 = bz.find_bz_lattice_vecs(a1, a2)
    # vtx, vty = bz.find_bz_zone(b1, b2)
    # print 'a1, a2 = ', (a1, a2)
    # print 'b1, b2 = ', (b1, b2)
    # print 'vtx, vty = ', (vtx, vty)
    # sys.exit()

    mM = calc_matrix(angs, num_neighbors, bls, tvals, ons, magnetic=True)

    fn = 'honeycomb_sheared'

    fn += '_%0.2f__' % t1234[0]
    fn += '%0.2f_' % ons[0]
    fn += '%0.2f__' % ons[1]
    fn += '%0.2f_' % (delta * 180 / pi)
    fn += '%0.2f' % (phi * 180 / pi)

    od = base_dir + '/honeycomb_sheared/'
    # print 'od is', od
    if not os.path.isdir(od):
        os.mkdir(od)
    if not os.path.isdir(od + 'data/'):
        os.mkdir(od + 'data/')
        os.mkdir(od + 'images/')

    return mM, vertex_points, od, fn, [1, 2]


def honeycomb_sheared_diff_bonds(t1234, d, p, ons, base_dir='/Users/lisa/Dropbox/Research/New_Chern_general/data/'):
    """Initiates a distorted and sheared honeycomb lattice.
    
    Parameters
    ----------
    t1234 : array of length 1
    
        The value for Omega_k 
        
    d : float
        The value of the angle delta in degrees
        
    p : float
        The value of the angle phi in degrees

    ons: array of length 2
        The on-site Omega_g value for the A and B sites respectively
        
    Returns
    ----------
    mM : function of two variables
        Function which calculates the interaction matrix for this lattice

    vertex_points : array of dimension 6x2
        points which trace out the boundary of the Brillouin zone

    od : string
        output directory for data based on lattice type
        
    fn : string
        file name

    [1, 2] : array
            particular to this lattice.  Number of different bond strengths and on-site energies respectively

    
    """
    t1 = t1234[0]

    if len(t1234) == 3:
        t2 = t1234[1]
        t3 = t1234[2]
    else:
        t2 = t3 = t1

    delta = d * pi / 180
    phi = p * pi / 180.
    theta = 0.5 * (2 * pi - delta) - pi / 2

    nA = array([[cos(theta), sin(theta)], [-cos(theta), sin(theta)], [-sin(phi), -cos(phi)]])
    nB = array([[cos(theta), -sin(theta)], [-cos(theta), -sin(theta)], [sin(phi), cos(phi)]])

    A = [0, 0]
    B = nA[2]

    angs = [[arctan2(nA[0, 1], nA[0, 0]), arctan2(nA[1, 1], nA[1, 0]), arctan2(nA[2, 1], nA[2, 0])],
            [arctan2(nB[0, 1], nB[0, 0]), arctan2(nB[1, 1], nB[1, 0]), arctan2(nB[2, 1], nB[2, 0])]]  # B (L) A (R)

    bls = [[1, 1, 1], [1, 1, 1]]

    num_neighbors = [[0, 3],
                     [3, 0]]

    tvals = [[t1, t2, t3],
             [t2, t1, t3]]

    a1, a2 = bz.find_lattice_vecs(delta, phi)
    b1, b2 = bz.find_bz_lattice_vecs(a1, a2)

    vtx, vty = bz.find_bz_zone(b1, b2)
    vertex_points = array([vtx, vty]).T

    mM = calc_matrix(angs, num_neighbors, bls, tvals, ons)

    fn = 'honeycomb_sheared_diff_bonds'
    od = base_dir + '/' + fn + '/'
    fn += '_%0.2f_' % t1234[0]
    fn += '_%0.2f_' % t1234[1]
    fn += '_%0.2f__' % t1234[2]
    fn += '%0.2f_' % ons[0]
    fn += '%0.2f__' % ons[1]
    fn += '%0.2f_' % (delta * 180 / pi)
    fn += '%0.2f' % (phi * 180 / pi)

    print 'od is', od
    if not os.path.isdir(od):
        os.mkdir(od)
    if not os.path.isdir(od + 'data/'):
        os.mkdir(od + 'data/')
        os.mkdir(od + 'images/')

    return mM, vertex_points, od, fn, [1, 2]


def kagome_lattice(t1234, ons, x1, x2, x3, z, base_dir='/Users/lisa/Dropbox/Research/New_Chern_general/data/',
                   magnetic=False):
    """Initiates a kagome lattice
    
    Parameters
    ----------
    t1234 : array of length 1
        The 4 values for Omega_k of the different springs.  
    
    ons: array of length 4
        The on-site Omega_g value for the A, B, C, & D sites respectively
        
    Returns
    ----------
    mM : function of two variables
        Function which calculates the interaction matrix for this lattice
    
    vertex_points : array of dimension 6x2
        points which trace out the boundary of the Brillouin zone
    
    od : string
        output directory for data based on lattice type
            
    fn : string
        file name
    
    
    [3, 1] : array
        particular to this lattice.  Number of different bond strengths and on-site energies.
    """
    t1 = t1234[0]

    pin = 1

    a = 2 * array([[cos(2 * pi * p / 3.), sin(2 * pi * p / 3.), 0] for p in range(3)])
    a1 = a[0];
    a2 = a[1];
    a3 = a[2];

    # make unit cell
    x = array([x1, x2, x3]);
    y = [z / 3. + x[mod(p - 1, 3)] - x[mod(p + 1, 3)] for p in range(3)]
    s = array([x[p] * (a[mod(p - 1, 3)] - a[mod(p + 1, 3)]) + y[p] * a[p] for p in range(3)])
    s1 = s[0];
    s2 = s[1];
    d1 = a1 / 2. + s2
    d2 = a2 / 2. - s1
    d3 = a3 / 2.

    # nodes at R (from top to bottom) -- like fig 2a of KaneLubensky2014

    # R = array([d1+a2,
    # d3+a1+a2,
    # d2+a1,
    # d1,
    # d3+a1,
    # d2-a2,
    # d1+a3,
    # d3,
    # d2+a3,
    # d1+a2+a3,
    # d3+a2,
    # d2])
    #


    nA = array([
        d2 + a1,
        d2 - a2,
        d3 + a1,
        d3 + a1 + a2
    ]) - d1

    nB = array([d1 + a2,
                d1 + a2 + a3,
                d3 + a2,
                d3 + a1 + a2
                ]) - d2

    nC = array([d1 + a3,
                d1 + a2 + a3,
                d2 - a2,
                d2 + a3
                ]) - d3

    g = lambda arr: [arctan2(arr[i, 1], arr[i, 0]) for i in range(len(arr))]
    f = lambda arr: [sqrt(arr[i, 0] ** 2 + arr[i, 1] ** 2) for i in range(len(arr))]

    print 'drawing points...'

    angs = [g(nA), g(nB), g(nC)]

    print 'angs'
    print array(angs) * 180. / pi

    bls = [f(nA), f(nB), f(nC)]

    print 'bls', bls

    tvals = [[t1, t1, t1, t1],
             [t1, t1, t1, t1],
             [t1, t1, t1, t1]]

    num_neighbors = [[0, 2, 2],
                     [2, 0, 2],
                     [2, 2, 0],

                     ]

    b1, b2 = bz.find_bz_lattice_vecs(a3, a1)
    vtx, vty = bz.find_bz_zone(b1, b2)
    vertex_points = array([vtx, vty]).T

    mM = calc_matrix(angs, num_neighbors, bls=bls, tvals=tvals, ons=ons, magnetic=magnetic)

    fn = 'kagome_lattice_'

    fn += '_%0.2f__' % t1234[0]

    fn += '%0.2f_' % ons[0]
    fn += '%0.2f_' % ons[1]
    fn += '%0.2f_' % ons[2]

    fn += '%0.2f_' % x1
    fn += '%0.2f_' % x2
    fn += '%0.2f_' % x3
    fn += '%0.2f_' % z

    if magnetic:

        od = base_dir + '/kagome_magnetic_lattice/'
    else:
        od = base_dir + '/kagome_lattice/'
    if not os.path.isdir(od):
        os.mkdir(od)
        os.mkdir(od + 'data/')
        os.mkdir(od + 'images/')
    return mM, vertex_points, od, fn, [1, 3]


def honeycomb_sheared_vis(d, p):
    """Makes data for picture of sheared honeycomb lattice
    
    Parameters
    ----------
    d : float
        Value of delta for the lattice in degrees

    p : float
        Value of phi for the lattice in degrees
        
    Returns
    ----------
    R : matrix of dimension nx3
        Equilibrium positions of all the gyroscopes 
        
    Ni : matrix of dimension n x (max number of neighbors)
        Each row corresponds to a gyroscope.  The entries tell the numbers of the neighboring gyroscopes 
        
    Nk : matrix of dimension n x (max number of neighbors)
        Correponds to Ni matrix.  1 corresponds to a true connection while 0 signifies that there is not a connection
        
    cols : array of length # sites
        colors for the sites in the plot
    
    lincols : array of length # bonds
        colors for the bonds in the plot
    """

    pin = 1
    deltaz = 1

    delta = d * pi / 180
    phi = p * pi / 180.
    theta = 0.5 * (2 * pi - delta) - pi / 2

    nA = array([[cos(theta), sin(theta)], [-cos(theta), sin(theta)], [-sin(phi), -cos(phi)]])
    nB = array([[cos(theta), -sin(theta)], [-cos(theta), -sin(theta)], [sin(phi), cos(phi)]])

    A = [0, 0]
    B = nA[2]

    R = [A,
         A + nA[0],
         A + nA[1],
         B,
         B + nB[0],
         B + nB[1],
         A + nA[1] + nB[1],
         A + nA[1] + nB[1] + nA[2],
         A + nA[0] + nB[0],
         A + nA[0] + nB[0] + nA[2]

         ]

    Ni = [[1, 2, 3],
          [0, 8, 0],
          [0, 6, 0],
          [0, 4, 5],
          [3, 9, 0],
          [3, 7, 0],
          [2, 7, 0],
          [6, 5, 0],
          [1, 9, 0],
          [8, 4, 0]
          ]

    Nk = [[1, 1, 1],
          [1, 1, 0],
          [1, 1, 0],
          [1, 1, 1],
          [1, 1, 0],
          [1, 1, 0],
          [1, 1, 0],
          [1, 1, 0],
          [1, 1, 0],
          [1, 1, 0]
          ]

    c1 = '#000000'
    c2 = '#B9B9B9'
    c3 = '#E51B1B'
    c4 = '#18CFCF'

    col = [c1, c2, c3, c4]

    A = col[0]
    B = col[1]
    C = col[2]
    D = col[3]

    cc1 = '#96adb4'
    cols = array([A, B, B, B, A, A, A, B, A, B])
    line_cols = array([cc1, cc1, cc1, cc1, cc1, cc1, cc1, cc1, cc1, cc1, cc1])

    return array(R), array(Ni), array(Nk), cols, line_cols


def honeycomb_sheared_vis_NNN(d, p):
    """Makes data for picture of sheared honeycomb lattice
    
    Parameters
    ----------
    d : float
        Value of delta for the lattice in degrees

    p : float
        Value of phi for the lattice in degrees
        
    Returns
    ----------
    R : matrix of dimension nx3
        Equilibrium positions of all the gyroscopes 
        
    Ni : matrix of dimension n x (max number of neighbors)
        Each row corresponds to a gyroscope.  The entries tell the numbers of the neighboring gyroscopes 
        
    Nk : matrix of dimension n x (max number of neighbors)
        Correponds to Ni matrix.  1 corresponds to a true connection while 0 signifies that there is not a connection
        
    cols : array of length # sites
        colors for the sites in the plot
    
    lincols : array of length # bonds
        colors for the bonds in the plot
    """

    pin = 1
    deltaz = 1

    delta = d * pi / 180
    phi = p * pi / 180.
    theta = 0.5 * (2 * pi - delta) - pi / 2

    nA = array([[cos(theta), sin(theta)], [-cos(theta), sin(theta)], [-sin(phi), -cos(phi)]])
    nB = array([[cos(theta), -sin(theta)], [-cos(theta), -sin(theta)], [sin(phi), cos(phi)]])

    A = [0, 0]
    B = nA[2]

    R = [A,  # 0
         A + nA[0],  # 1
         A + nA[1],  # 2
         B,  # 3
         B + nB[0],  # 4
         B + nB[1],  # 5
         A + nA[1] + nB[1],  # 6
         A + nA[1] + nB[1] + nA[2],  # 7
         A + nA[0] + nB[0],  # 8
         A + nA[0] + nB[0] + nA[2]

         ]

    Ni = [[1, 2, 3, ],
          [0, 8, 0],
          [0, 6, 0],
          [0, 4, 5],
          [3, 9, 0],
          [3, 7, 0],
          [2, 7, 0],
          [6, 5, 0],
          [1, 9, 0],
          [8, 4, 0]
          ]

    Nk = [[1, 1, 1],
          [1, 1, 0],
          [1, 1, 0],
          [1, 1, 1],
          [1, 1, 0],
          [1, 1, 0],
          [1, 1, 0],
          [1, 1, 0],
          [1, 1, 0],
          [1, 1, 0]
          ]

    c1 = '#000000'
    c2 = '#B9B9B9'
    c3 = '#E51B1B'
    c4 = '#18CFCF'

    col = [c1, c2, c3, c4]

    A = col[0]
    B = col[1]
    C = col[2]
    D = col[3]

    cc1 = '#96adb4'
    cols = array([A, B, B, B, A, A, A, B, A, B])
    line_cols = array([cc1, cc1, cc1, cc1, cc1, cc1, cc1, cc1, cc1, cc1, cc1])

    return array(R), array(Ni), array(Nk), cols, line_cols


def alpha_vis():
    """Makes data for picture of alpha lattice
    
    Returns
    ----------
    R : matrix of dimension nx3
        Equilibrium positions of all the gyroscopes 
        
    Ni : matrix of dimension n x (max number of neighbors)
        Each row corresponds to a gyroscope.  The entries tell the numbers of the neighboring gyroscopes 
        
    Nk : matrix of dimension n x (max number of neighbors)
        Correponds to Ni matrix.  1 corresponds to a true connection while 0 signifies that there is not a connection
        
    cols : array of length # sites
        colors for the sites in the plot
    
    lincols : array of length # bonds
        colors for the bonds in the plot
    """
    th = 60 * pi / 180
    nA = array([[0, 1], [0, -1], [-1 * cos(th / 2), 1 * sin(th / 2)], [-1 * cos(th / 2), -1 * sin(th / 2)]])  # B B C D
    nB = array([[0, 1], [0, -1], [1 * cos(th / 2), -1 * sin(th / 2)], [1 * cos(th / 2), 1 * sin(th / 2)]])  # A A C D
    nC = array([[1 * cos(th / 2), -1 * sin(th / 2)], [-1 * cos(th / 2), 1 * sin(th / 2)]])  # A B
    nD = array([[1 * cos(th / 2), 1 * sin(th / 2)], [-1 * cos(th / 2), -1 * sin(th / 2)]])  # A B

    c1 = '#000000'
    c2 = '#FFFFFF'
    c3 = '#E51B1B'
    c4 = '#18CFCF'

    cc1 = '#96adb4'
    cc2 = '#b16566'
    cc3 = '#665656'
    cc4 = '#9d96b4'

    col = [c1, c2, c3, c4]
    lincol = [cc1, cc2, cc3, cc4]

    Apos = array([0, 0])
    Bpos = array([0, -1])
    R = array([Apos,  # A-0
               Bpos,  # B-1
               Apos + nA[0],  # B-2
               Apos + nA[2],  # C-3
               Apos + nA[3],  # D-4
               Bpos + nB[1],  # A-5
               Bpos + nB[2],  # C-6
               Bpos + nB[3]  # D-7

               ])
    A = col[0]
    B = col[1]
    C = col[2]
    D = col[3]

    c1 = lincol[0]
    c2 = lincol[1]
    c3 = lincol[2]
    c4 = lincol[3]

    cols = array([A, B, B, C, D, A, C, D])
    line_cols = array([c1, c2, c4, c3, c2, c4, c3])
    Ni = array([[1, 2, 3, 4],  # 0
                [0, 5, 6, 7],  # 1
                [0, -0, -0, -0],  # 2
                [0, -0, -0, -0],  # 3
                [0, -0, -0, -0],  # 4
                [1, -0, -0, -0],  # 5
                [1, -0, -0, -0],  # 6
                [1, -0, -0, -0]
                ])

    Nk = array([[1, 1, 1, 1],  # 0
                [1, 1, 1, 1],  # 1
                [1, -0, -0, -0],  # 2
                [1, -0, -0, -0],  # 3
                [1, -0, -0, -0],  # 4
                [1, -0, -0, -0],  # 5
                [1, -0, -0, -0],  # 6
                [1, -0, -0, -0]
                ])

    return array(R), array(Ni), array(Nk), cols, line_cols


def kagome_vis(x1, x2, x3, z):
    """Makes data for picture of kagome lattice
    
    Parameters
    ----------
    d : float
        Value of delta for the lattice in degrees

    p : float
        Value of phi for the lattice in degrees
        
    Returns
    ----------
    R : matrix of dimension nx3
        Equilibrium positions of all the gyroscopes 
        
    Ni : matrix of dimension n x (max number of neighbors)
        Each row corresponds to a gyroscope.  The entries tell the numbers of the neighboring gyroscopes 
        
    Nk : matrix of dimension n x (max number of neighbors)
        Correponds to Ni matrix.  1 corresponds to a true connection while 0 signifies that there is not a connection
        
    cols : array of length # sites
        colors for the sites in the plot
    
    lincols : array of length # bonds
        colors for the bonds in the plot
    """

    pin = 1
    deltaz = 1
    # Bravais primitive unit vecs
    a = 2 * array([[cos(2 * pi * p / 3.), sin(2 * pi * p / 3.)] for p in range(3)])
    a1 = a[0];
    a2 = a[1];
    a3 = a[2];

    # make unit cell
    x = array([x1, x2, x3])
    y = [z / 3. + x[mod(p - 1, 3)] - x[mod(p + 1, 3)] for p in range(3)]
    s = array([x[p] * (a[mod(p - 1, 3)] - a[mod(p + 1, 3)]) + y[p] * a[p] for p in range(3)])
    s1 = s[0];
    s2 = s[1];
    d1 = a1 / 2. + s2
    d2 = a2 / 2. - s1
    d3 = a3 / 2.

    # nodes at R (from top to bottom) -- like fig 2a of KaneLubensky2014
    R = array([d1 + a2,
               d3 + a1 + a2,
               d2 + a1,
               d1,
               d3 + a1,
               d2 - a2,
               d1 + a3,
               d3,
               d2 + a3,
               d1 + a2 + a3,
               d3 + a2,
               d2])

    Ni = array([
        [11, 1, 0, 0],  # 0 d1+a2,
        [3, 11, 0, 2],  # 1 d3+a1+a2,
        [3, 1, 0, 0],  # 2 d2+a1,
        [2, 5, 4, 1],  # 3 d1,
        [3, 5, 0, 0],  # 4 d3+a1,
        [3, 7, 4, 6],  # 5 d2-a2,
        [7, 5, 0, 0],  # 6 d1+a3,
        [6, 9, 5, 8],  # 7 d3,
        [7, 9, 0, 0],  # 8 d2+a3,
        [11, 7, 10, 8],  # 9 d1+a2+a3,
        [11, 9, 0, 0],  # 10 d3+a2,
        [0, 9, 10, 1],  # 11 d2
    ])

    Nk = array([
        [1, 1, 0, 0],  # 0 d1+a2,
        [1, 1, 1, 1],  # 1 d3+a1+a2,
        [1, 1, 0, 0],  # 2 d2+a1,
        [1, 1, 1, 1],  # 3 d1,
        [1, 1, 0, 0],  # 4 d3+a1,
        [1, 1, 1, 1],  # 5 d2-a2,
        [1, 1, 0, 0],  # 6 d1+a3,
        [1, 1, 1, 1],  # 7 d3,
        [1, 1, 0, 0],  # 8 d2+a3,
        [1, 1, 1, 1],  # 9 d1+a2+a3,
        [1, 1, 0, 0],  # 10 d3+a2,
        [1, 1, 1, 1],  # 11 d2
    ])

    c1 = '#000000'
    c2 = '#B9B9B9'
    c3 = '#E51B1B'
    c4 = '#18CFCF'

    col = [c1, c2, c3, c4]

    A = col[0]
    B = col[1]
    C = col[2]

    cc1 = '#96adb4'
    cols = array([A, C, B, A, C, B, A, C, B, A, C, B])
    line_cols = array([cc1, cc1, cc1, cc1, cc1, cc1, cc1, cc1, cc1, cc1, cc1, cc1, cc1, cc1, cc1, cc1, cc1, cc1])

    return array(R), array(Ni), array(Nk), cols, line_cols


def lattice_plot(R, Ni, Nk, ax, colors, line_cols, opt=False):
    """draws lines for the gyro lattice (white lines connecting points)
    
    Parameters
    ----------
    R : matrix of dimension nx3
        Equilibrium positions of all the gyroscopes 
            
    Ni : matrix of dimension n x (max number of neighbors)
            Each row corresponds to a gyroscope.  The entries tell the numbers of the neighboring gyroscopes 
            
    Nk : matrix of dimension n x (max number of neighbors)
            Correponds to Ni matrix.  1 corresponds to a true connection while 0 signifies that there is not a connection
    
    ax: python axis 
        axis on which to draw lattice
        
    #    """
    Rx = 1. * R[:, 0]
    Ry = 1. * R[:, 1]
    R_p = R.copy()

    plt.sca(ax)
    ax.set_aspect('equal')
    # ax.set_axis_bgcolor('#E8E8E8')


    pylab.xlim(Rx.min() - 1, Rx.max() + 1)
    pylab.ylim(Ry.min() - 1, Ry.max() + 1)
    ax.set_autoscale_on(False)

    sl = abs((Ry.min() - Ry.max()) / 2)

    # points, h = outer_hexagon(sl+1.)

    ppu = get_points_per_unit()
    s = absolute_sizer()

    CR = 0.4
    LW = 0.25  # 0.15
    patch = []
    NP = len(Ni)
    ln = 0

    for i in range(NP):
        for j, k in zip(Ni[i], Nk[i]):
            if i < j and k > 0:
                ax.plot(R_p[(i, j), 0], R_p[(i, j), 1], line_cols[ln], linewidth=0.6, zorder=0)
                ln = ln + 1
                circ = Circle((Rx[i], Ry[i]), radius=0.25)

    for i in range(NP):
        circ = Circle((Rx[i], Ry[i]), radius=0.2, linewidth=0.6)
        patch.append(circ)

    p = PatchCollection(patch, facecolors=colors, edgecolors='k')
    ax.add_collection(p)


def get_points_per_unit(ax=None):
    if ax is None: ax = pylab.gca()
    ax.apply_aspect()
    x0, x1 = ax.get_xlim()
    return ax.bbox.width / abs(x1 - x0)


def absolute_sizer(ax=None):
    ppu = get_points_per_unit(ax)
    return lambda x: pi * (x * ppu) ** 2


def honeycomb_sheared_semi_infinite_test(t1234, d, p, ons, ny,
                                         base_dir='/Users/lisa/Dropbox/Research/New_Chern_general/data/'):
    """Initiates a distorted and sheared honeycomb lattice.  In semi-infinite geometry
    
    Parameters
    ----------
    t1234 : array of length 1
        The value for Omega_k 
        
    d : float
        The value of the angle delta in degrees
        
    p : float
        The value of the angle phi in degrees

    ons: array of length ny
        The on-site Omega_g value for the A and B sites respectively
        
    Returns
    ----------
    mM : function of two variables
        Function which calculates the interaction matrix for this lattice

    vertex_points : array of dimension 6x2
        points which trace out the boundary of the Brillouin zone

    od : string
        output directory for data based on lattice type
        
    fn : string
        file name

    [1, 2] : array
            particular to this lattice.  Number of different bond strengths and on-site energies respectively

    
    """
    pin = 1
    deltaz = 1

    t1 = float(t1234[0])
    print 't1 is', t1
    delta = d * pi / 180
    phi = p * pi / 180.
    theta = 0.5 * (2 * pi - delta) - pi / 2

    nB = array([[cos(theta), sin(theta)], [-cos(theta), sin(theta)], [-sin(phi), -cos(phi)]])
    nA = array([[sin(phi), cos(phi)], [cos(theta), -sin(theta)], [-cos(theta), -sin(theta)]])

    angsA = [arctan2(nA[i, 1], nA[i, 0]) for i in range(len(nA))]
    angsB = [arctan2(nB[i, 1], nB[i, 0]) for i in range(len(nB))]

    angsAB = [angsA, angsB]
    angs = [angsAB[i % 2] for i in range(ny)]
    angs[0] = [angsA[2], angsA[1], angsA[0]]

    bls = list(ones_like(angs))
    tvals = list(t1 * ones_like(angs))

    num_neighbors = zeros((ny, ny), dtype=int)
    for i in range(ny):
        if (i != 0 and i != ny - 1):
            if i % 2 == 0:
                num_neighbors[i, i + 1] = 2
                num_neighbors[i, i - 1] = 1
            else:
                num_neighbors[i, i - 1] = 2
                num_neighbors[i, i + 1] = 1
        else:
            if i == 0:
                num_neighbors[i, i + 1] = 2
            else:
                if i % 2 == 0:
                    num_neighbors[i, i - 1] = 1
                else:
                    num_neighbors[i, i - 1] = 2

    a1, a2 = bz.find_lattice_vecs(delta, phi)
    b1, b2 = bz.find_bz_lattice_vecs(a1, a2)

    vtx, vty = bz.find_bz_zone(b1, b2)
    vertex_points = array([vtx, vty]).T

    mM = calc_matrix(angs, num_neighbors, bls, tvals, ons)

    fn = 'honeycomb_sheared_strip_%03d' % ny

    fn += '_%0.2f__' % t1234[0]
    fn += '%0.2f_' % ons[0]
    fn += '%0.2f__' % ons[1]
    fn += '%0.2f_' % (delta * 180 / pi)
    fn += '%0.2f' % (phi * 180 / pi)

    od = base_dir + '/honeycomb_sheared_strip_%03d/' % ny
    if not os.path.isdir(od):
        os.mkdir(od)
        os.mkdir(od + 'data/')
        os.mkdir(od + 'images/')

    return mM, vertex_points, od, fn, [1, 2]


def kagome_lattice_strip(t1234, ons, x1, x2, x3, z, ny,
                         base_dir='/Users/lisa/Dropbox/Research/New_Chern_general/data/'):
    """Initiates a kagome lattice
    
    Parameters
    ----------
    t1234 : array of length 1
        The 4 values for Omega_k of the different springs.  
    
    ons: array of length 4
        The on-site Omega_g value for the A, B, C, & D sites respectively
        
    Returns
    ----------
    mM : function of two variables
        Function which calculates the interaction matrix for this lattice
    
    vertex_points : array of dimension 6x2
        points which trace out the boundary of the Brillouin zone
    
    od : string
        output directory for data based on lattice type
            
    fn : string
        file name
    
    
    [3, 1] : array
        particular to this lattice.  Number of different bond strengths and on-site energies.
    """
    t1 = t1234[0]

    pin = 1

    a = 2 * array([[cos(2 * pi * p / 3.), sin(2 * pi * p / 3.), 0] for p in range(3)])
    a1 = a[0];
    a2 = a[1];
    a3 = a[2];

    # make unit cell
    x = array([x1, x2, x3])
    y = [z / 3. + x[mod(p - 1, 3)] - x[mod(p + 1, 3)] for p in range(3)]
    s = array([x[p] * (a[mod(p - 1, 3)] - a[mod(p + 1, 3)]) + y[p] * a[p] for p in range(3)])
    s1 = s[0];
    s2 = s[1];
    d1 = a1 / 2. + s2
    d2 = a2 / 2. - s1
    d3 = a3 / 2.

    # nodes at R (from top to bottom) -- like fig 2a of KaneLubensky2014

    nA = array([
        d2 + a1,
        d2 - a2,
        d3 + a1,
        d3 + a1 + a2
    ]) - d1

    nB = array([d1 + a2,
                d1 + a2 + a3,
                d3 + a2,
                d3 + a1 + a2
                ]) - d2

    nC = array([d1 + a3,
                d1 + a2 + a3,
                d2 - a2,
                d2 + a3
                ]) - d3

    g = lambda arr: [arctan2(arr[i, 1], arr[i, 0]) for i in range(len(arr))]
    f = lambda arr: [sqrt(arr[i, 0] ** 2 + arr[i, 1] ** 2) for i in range(len(arr))]

    angs = [g(nA), g(nB), g(nC)]

    bls = [f(nA), f(nB), f(nC)]

    tvals = [[t1, t1, t1, t1],
             [t1, t1, t1, t1],
             [t1, t1, t1, t1]]

    angsA = angs[0]
    angsB = angs[1]
    angsC = angs[2]
    si = argsort(angsA)

    n0 = nA[[0, 3, 2, 1]]
    n1 = nC[[1, 2, 3, 0]]
    n2 = nB[[0, 2, 3, 1]]
    n3 = nA[[3, 0, 1, 2]]
    n4 = nB[[0, 3, 2, 1]]
    n5 = nC[[1, 2, 3, 0]]

    angsG = [[angsA[0], angsA[3], angsA[2], angsA[1]],
             [angsC[1], angsC[2], angsC[3], angsC[0]],
             [angsB[0], angsB[2], angsB[3], angsB[1]],
             [angsA[3], angsA[0], angsA[1], angsA[2]],
             [angsB[0], angsB[3], angsB[2], angsB[1]],
             [angsC[1], angsC[2], angsC[3], angsC[0]]
             ]

    blG = [f(n0), f(n1), f(n2), f(n3), f(n4), f(n5)]
    angsG = [g(n0), g(n1), g(n2), g(n3), g(n4), g(n5)]

    angs = [angsG[i % 6] for i in range(ny)]
    bls = [blG[i % 6] for i in range(ny)]

    tvals = list(t1 * ones_like(angs))

    num_neighbors = zeros((ny, ny), dtype=int)
    for i in range(ny):
        if (i != 0 and i < ny - 2):
            if i % 6 == 0:
                num_neighbors[i, i - 2] = 1
                num_neighbors[i, i - 1] = 1
                num_neighbors[i, i + 1] = 1
                num_neighbors[i, i + 2] = 1

            elif i % 6 == 1:

                num_neighbors[i, i - 1] = 1
                num_neighbors[i, i + 1] = 2
                num_neighbors[i, i + 2] = 1
            elif i % 6 == 2:
                num_neighbors[i, i - 2] = 1
                num_neighbors[i, i - 1] = 2
                num_neighbors[i, i + 1] = 1

            elif i % 6 == 3:
                num_neighbors[i, i - 2] = 1
                num_neighbors[i, i - 1] = 1
                num_neighbors[i, i + 1] = 1
                num_neighbors[i, i + 2] = 1
            elif i % 6 == 4:

                num_neighbors[i, i - 1] = 1
                num_neighbors[i, i + 1] = 2
                num_neighbors[i, i + 2] = 1
            elif i % 6 == 5:
                num_neighbors[i, i - 2] = 1
                num_neighbors[i, i - 1] = 2
                num_neighbors[i, i + 1] = 1

        else:
            if i == 0:
                num_neighbors[i, i + 1] = 1
                num_neighbors[i, i + 2] = 1
            elif i == ny - 1:  # last
                if i % 6 == 0:
                    num_neighbors[i, i - 2] = 1
                    num_neighbors[i, i - 1] = 1


                elif i % 6 == 1:

                    num_neighbors[i, i - 1] = 1

                elif i % 6 == 2:
                    num_neighbors[i, i - 2] = 1
                    num_neighbors[i, i - 1] = 2


                elif i % 6 == 3:
                    num_neighbors[i, i - 2] = 1
                    num_neighbors[i, i - 1] = 1

                elif i % 6 == 4:

                    num_neighbors[i, i - 1] = 1

                elif i % 6 == 5:
                    num_neighbors[i, i - 2] = 1
                    num_neighbors[i, i - 1] = 2
            else:  # (next to last)
                if i % 6 == 0:
                    num_neighbors[i, i - 2] = 1
                    num_neighbors[i, i - 1] = 1
                    num_neighbors[i, i + 1] = 1


                elif i % 6 == 1:

                    num_neighbors[i, i - 1] = 1
                    num_neighbors[i, i + 1] = 2

                elif i % 6 == 2:
                    num_neighbors[i, i - 2] = 1
                    num_neighbors[i, i - 1] = 2
                    num_neighbors[i, i + 1] = 1

                elif i % 6 == 3:
                    num_neighbors[i, i - 2] = 1
                    num_neighbors[i, i - 1] = 1
                    num_neighbors[i, i + 1] = 1

                elif i % 6 == 4:

                    num_neighbors[i, i - 1] = 1
                    num_neighbors[i, i + 1] = 2

                elif i % 6 == 5:
                    num_neighbors[i, i - 2] = 1
                    num_neighbors[i, i - 1] = 2

    b1, b2 = bz.find_bz_lattice_vecs(a3, a1)
    vtx, vty = bz.find_bz_zone(b1, b2)
    vertex_points = array([vtx, vty]).T

    mM = calc_matrix(angs, num_neighbors, bls=bls, tvals=tvals, ons=ons)

    fn = 'kagome_lattice_strip'

    fn += '_%0.2f__' % t1234[0]

    fn += '%0.2f_' % ons[0]
    fn += '%0.2f_' % ons[1]
    fn += '%0.2f_' % ons[2]

    fn += '%0.2f_' % x1
    fn += '%0.2f_' % x2
    fn += '%0.2f_' % x3
    fn += '%0.2f_' % z

    od = base_dir + '/kagome_lattice_strip_%03d/' % ny
    if not os.path.isdir(od):
        os.mkdir(od)
        os.mkdir(od + 'data/')
        os.mkdir(od + 'images/')
    return mM, vertex_points, od, fn, [1, 3]


def alpha_lattice_strip(t1234, ons, ny, base_dir='/Users/lisa/Dropbox/Research/New_Chern_general/data/'):
    """This function creates an `alpha' lattice. (This is a lattice geometry).
    
    Parameters
    ----------
    t1234 : array of length 4
        The 4 values for Omega_k of the different springs.  

    ons: array of length 4
        The on-site Omega_g value for the A, B, C, & D sites respectively
        
    Returns
    ----------
    mM : function of two variables
        Function which calculates the interaction matrix for this lattice

    vertex_points : array of dimension 6x2
        points which trace out the boundary of the Brillouin zone

    od : string
        output directory for data based on lattice type
        
    fn : string
        file name

    [4, 4] : array
            particular to this lattice.  Number of different bond strengths and on-site energies.
    """

    print 'ons is', ons
    t1 = t1234[0]
    t2 = t1234[1]
    t3 = t1234[2]
    t4 = t1234[3]
    pin = 1
    th = 60 * pi / 180.
    nA = array([[0, 1], [0, -1], [-1 * cos(th / 2), 1 * sin(th / 2)], [-1 * cos(th / 2), -1 * sin(th / 2)]])  # B B C D
    nB = array([[0, 1], [0, -1], [1 * cos(th / 2), -1 * sin(th / 2)], [1 * cos(th / 2), 1 * sin(th / 2)]])  # A A C D
    nC = array([[1 * cos(th / 2), -1 * sin(th / 2)], [-1 * cos(th / 2), 1 * sin(th / 2)]])  # A B
    nD = array([[1 * cos(th / 2), 1 * sin(th / 2)], [-1 * cos(th / 2), -1 * sin(th / 2)]])  # A B

    g = lambda arr: [arctan2(arr[i, 1], arr[i, 0]) for i in range(len(arr))]
    f = lambda arr: [sqrt(arr[i, 0] ** 2 + arr[i, 1] ** 2) for i in range(len(arr))]

    tvals = array([[t2, t1, t4, t3],
                   [t1, t2, t4, t3],
                   [t4, t4],
                   [t3, t3]])

    n0 = nB[[0, 3, 2, 1]]
    n1 = nC[[1, 0]]
    n2 = nA[[0, 2, 3, 1]]
    n3 = nD[[0, 1]]

    t0 = array(tvals[1])[[0, 3, 2, 1]]
    t1 = array(tvals[2])[[1, 0]]
    t2 = array(tvals[0])[[0, 2, 3, 1]]
    t3 = array(tvals[3])[[0, 1]]

    blG = [f(n0), f(n1), f(n2), f(n3)]
    angsG = [g(n0), g(n1), g(n2), g(n3)]
    tvalsG = [t0, t1, t2, t3]

    angs = [angsG[i % 4] for i in range(ny)]
    bls = [blG[i % 4] for i in range(ny)]
    tvals = [tvalsG[i % 4] for i in range(ny)]
    angs[0] = array(angs[0])[[2, 3, 0, 1]]

    a1, a2 = bz.find_lattice_vecs_alpha_ns()
    b1, b2 = bz.find_bz_lattice_vecs(a1, a2)
    vtx, vty = bz.find_bz_zone_alpha(b1, b2)
    vertex_points = array([vtx, vty]).T

    num_neighbors = zeros((ny, ny), dtype=int)
    for i in range(ny):
        if (i != 0 and i < ny - 2):
            if i % 4 == 0:
                num_neighbors[i, i - 2] = 1
                num_neighbors[i, i - 1] = 1
                num_neighbors[i, i + 1] = 1
                num_neighbors[i, i + 2] = 1

            elif i % 4 == 1:
                num_neighbors[i, i - 1] = 1
                num_neighbors[i, i + 1] = 1

            elif i % 4 == 2:
                num_neighbors[i, i - 2] = 1
                num_neighbors[i, i - 1] = 1
                num_neighbors[i, i + 1] = 1
                num_neighbors[i, i + 2] = 1

            elif i % 4 == 3:
                num_neighbors[i, i - 1] = 1
                num_neighbors[i, i + 1] = 1

        else:
            if i == 0:
                num_neighbors[i, i + 1] = 1
                num_neighbors[i, i + 2] = 1
            elif i == ny - 1:  # last
                if i % 4 == 0:
                    num_neighbors[i, i - 2] = 1
                    num_neighbors[i, i - 1] = 1


                elif i % 4 == 1:
                    num_neighbors[i, i - 1] = 1


                elif i % 4 == 2:
                    num_neighbors[i, i - 2] = 1
                    num_neighbors[i, i - 1] = 1


                elif i % 4 == 3:
                    num_neighbors[i, i - 1] = 1
            else:  # next to last
                if i % 4 == 0:
                    num_neighbors[i, i - 2] = 1
                    num_neighbors[i, i - 1] = 1
                    num_neighbors[i, i + 1] = 1


                elif i % 4 == 1:
                    num_neighbors[i, i - 1] = 1
                    num_neighbors[i, i + 1] = 1

                elif i % 4 == 2:
                    num_neighbors[i, i - 2] = 1
                    num_neighbors[i, i - 1] = 1
                    num_neighbors[i, i + 1] = 1


                elif i % 4 == 3:
                    num_neighbors[i, i - 1] = 1
                    num_neighbors[i, i + 1] = 1

    print 'ons is', ons
    mM = calc_matrix(angs, num_neighbors, bls, tvals, ons)

    s1 = '%0.2f_' % t1234[0]
    s2 = '%0.2f_' % t1234[1]
    s3 = '%0.2f_' % t1234[2]
    s4 = '%0.2f_' % t1234[3]

    s5 = '%0.2f_' % ons[0]
    s6 = '%0.2f_' % ons[1]
    s7 = '%0.2f_' % ons[2]
    s8 = '%0.2f_' % ons[3]

    old_fn = 'data_dict' + s1 + s2 + s3 + s4 + '__' + s5 + s6 + s7 + s8

    fn = 'alpha_lattice_'

    fn += '_%0.2f_' % t1234[0]
    fn += '%0.2f_' % t1234[1]
    fn += '%0.2f_' % t1234[2]
    fn += '%0.2f__' % t1234[3]

    fn += '%0.2f_' % ons[0]
    fn += '%0.2f_' % ons[1]
    fn += '%0.2f_' % ons[2]
    fn += '%0.2f_' % ons[3]

    od = base_dir + '/alpha_lattice_strip/'
    if not os.path.isdir(od):
        os.mkdir(od)
        os.mkdir(od + 'data/')
        os.mkdir(od + 'images/')
    return mM, vertex_points, od, fn, [4, 4]


def alpha_lattice_strip_ky(t1234, ons, ny, base_dir='/Users/lisa/Dropbox/Research/New_Chern_general/data/'):
    """This function creates an `alpha' lattice. (This is a lattice geometry).
    
    Parameters
    ----------
    t1234 : array of length 4
        The 4 values for Omega_k of the different springs.  

    ons: array of length 4
        The on-site Omega_g value for the A, B, C, & D sites respectively
        
    Returns
    ----------
    mM : function of two variables
        Function which calculates the interaction matrix for this lattice

    vertex_points : array of dimension 6x2
        points which trace out the boundary of the Brillouin zone

    od : string
        output directory for data based on lattice type
        
    fn : string
        file name

    [4, 4] : array
            particular to this lattice.  Number of different bond strengths and on-site energies.
    """

    onsA = ons[0]
    onsB = ons[1]
    onsC = ons[2]
    onsD = ons[3]

    onsG = [onsA, onsB, onsC, onsD]

    print 'onsG', onsG

    ons = [onsG[i % 4] for i in range(ny)]

    t1 = t1234[0]
    t2 = t1234[1]
    t3 = t1234[2]
    t4 = t1234[3]
    pin = 1
    th = 60 * pi / 180.
    nA = array([[0, 1], [0, -1], [-1 * cos(th / 2), 1 * sin(th / 2)], [-1 * cos(th / 2), -1 * sin(th / 2)]])  # B B C D
    nB = array([[0, 1], [0, -1], [1 * cos(th / 2), -1 * sin(th / 2)], [1 * cos(th / 2), 1 * sin(th / 2)]])  # A A C D
    nC = array([[1 * cos(th / 2), -1 * sin(th / 2)], [-1 * cos(th / 2), 1 * sin(th / 2)]])  # A B
    nD = array([[1 * cos(th / 2), 1 * sin(th / 2)], [-1 * cos(th / 2), -1 * sin(th / 2)]])  # A B

    g = lambda arr: [arctan2(arr[i, 1], arr[i, 0]) for i in range(len(arr))]
    f = lambda arr: [sqrt(arr[i, 0] ** 2 + arr[i, 1] ** 2) for i in range(len(arr))]

    tvals = array([[t2, t1, t4, t3],
                   [t1, t2, t4, t3],
                   [t4, t4],
                   [t3, t3]])

    n0 = nA[[3, 2, 0, 1]]
    n1 = nB[[0, 1, 3, 2]]
    n2 = nD[[1, 0]]
    n3 = nC[[1, 0]]

    t0 = array(tvals[0])[[3, 2, 0, 1]]
    t1 = array(tvals[1])[[0, 1, 3, 2]]
    t2 = array(tvals[3])[[1, 0]]
    t3 = array(tvals[2])[[1, 0]]

    blG = [f(n0), f(n1), f(n2), f(n3)]
    angsG = [g(n0), g(n1), g(n2), g(n3)]
    tvalsG = [t0, t1, t2, t3]

    angs = [angsG[i % 4] for i in range(ny)]
    bls = [blG[i % 4] for i in range(ny)]
    tvals = [tvalsG[i % 4] for i in range(ny)]
    angs[0] = array(angs[0])[[2, 3, 0, 1]]

    a1, a2 = bz.find_lattice_vecs_alpha_ns()
    b1, b2 = bz.find_bz_lattice_vecs(a1, a2)
    vtx, vty = bz.find_bz_zone_alpha(b1, b2)
    vertex_points = array([vtx, vty]).T

    num_neighbors = zeros((ny, ny), dtype=int)
    for i in range(ny):
        if (i != 0 and i < ny - 2):
            if i % 4 == 0:
                num_neighbors[i, i - 2] = 1
                num_neighbors[i, i - 1] = 1
                num_neighbors[i, i + 1] = 2


            elif i % 4 == 1:
                num_neighbors[i, i - 1] = 2
                num_neighbors[i, i + 1] = 1
                num_neighbors[i, i + 2] = 1

            elif i % 4 == 2:
                num_neighbors[i, i - 1] = 1
                num_neighbors[i, i + 2] = 1


            elif i % 4 == 3:
                num_neighbors[i, i - 2] = 1
                num_neighbors[i, i + 1] = 1

        else:
            if i == 0:
                num_neighbors[i, i + 1] = 2

            elif i == ny - 1:  # last
                if i % 4 == 0:
                    num_neighbors[i, i - 2] = 1
                    num_neighbors[i, i - 1] = 1

                elif i % 4 == 1:
                    num_neighbors[i, i - 1] = 2

                elif i % 4 == 2:
                    num_neighbors[i, i - 1] = 1



                elif i % 4 == 3:
                    num_neighbors[i, i - 2] = 1

            else:  # next to last
                if i % 4 == 0:
                    num_neighbors[i, i - 2] = 1
                    num_neighbors[i, i - 1] = 1
                    num_neighbors[i, i + 1] = 2


                elif i % 4 == 1:
                    num_neighbors[i, i - 1] = 2
                    num_neighbors[i, i + 1] = 1

                elif i % 4 == 2:
                    num_neighbors[i, i - 1] = 1



                elif i % 4 == 3:
                    num_neighbors[i, i - 2] = 1
                    num_neighbors[i, i + 1] = 1

    mM = calc_matrix(angs, num_neighbors, bls, tvals, ons)

    s1 = '%0.2f_' % t1234[0]
    s2 = '%0.2f_' % t1234[1]
    s3 = '%0.2f_' % t1234[2]
    s4 = '%0.2f_' % t1234[3]

    s5 = '%0.2f_' % ons[0]
    s6 = '%0.2f_' % ons[1]
    s7 = '%0.2f_' % ons[2]
    s8 = '%0.2f_' % ons[3]

    old_fn = 'data_dict' + s1 + s2 + s3 + s4 + '__' + s5 + s6 + s7 + s8

    fn = 'alpha_lattice_'

    fn += '_%0.2f_' % t1234[0]
    fn += '%0.2f_' % t1234[1]
    fn += '%0.2f_' % t1234[2]
    fn += '%0.2f__' % t1234[3]

    fn += '%0.2f_' % ons[0]
    fn += '%0.2f_' % ons[1]
    fn += '%0.2f_' % ons[2]
    fn += '%0.2f_' % ons[3]

    od = base_dir + '/alpha_lattice_strip_ky_%03d/' % ny
    if not os.path.isdir(od):
        os.mkdir(od)
        os.mkdir(od + 'data/')
        os.mkdir(od + 'images/')
    return mM, vertex_points, od, fn, [4, 4]


def kagome_lattice_strip_new_lv(t1234, ons, alpha, ny, base_dir='/Users/lisa/Dropbox/Research/New_Chern_general/data/'):
    """Initiates a kagome lattice
    
    Parameters
    ----------
    t1234 : array of length 1
        The 4 values for Omega_k of the different springs.  
    
    ons: array of length 4
        The on-site Omega_g value for the A, B, C, & D sites respectively
        
    Returns
    ----------
    mM : function of two variables
        Function which calculates the interaction matrix for this lattice
    
    vertex_points : array of dimension 6x2
        points which trace out the boundary of the Brillouin zone
    
    od : string
        output directory for data based on lattice type
            
    fn : string
        file name
    
    
    [3, 1] : array
        particular to this lattice.  Number of different bond strengths and on-site energies.
    """
    t1 = t1234[0]

    alpha = -alpha
    pin = 1
    deltaz = 1
    # Bravais primitive unit vecs
    print 'cos alpha', cos(alpha)
    a = 2 * array([[cos(2 * pi * p / 3.), sin(2 * pi * p / 3.), 0] for p in range(3)]) * (cos(alpha))
    a1 = a[0];
    a2 = a[1];
    a3 = a[2];

    # make unit cell

    psi_n = [(n - 1) * pi / 3 + (-1) ** n * alpha for n in range(7)]
    print 'psi n', psi_n[1:]
    b = array([cos(psi_n), sin(psi_n), zeros_like(psi_n)]).T
    b = b[1:]
    print 'b', b

    points = zeros_like(b)
    for i in range(len(b)):
        if i > 0:
            print i, b[i - 1]
            points[i] = points[i - 1] + b[i - 1]
    # fig = plt.figure()
    # plt.scatter(points[:,0], points[:,1])
    # plt.gca().set_aspect(1)
    # plt.show()
    print 'points', points
    d1 = points[2]
    d2 = points[4]
    d3 = points[0]

    # nodes at R (from top to bottom) -- like fig 2a of KaneLubensky2014

    nA = array([
        d2 + a1,
        d2 - a2,
        d3 + a1,
        d3 + a1 + a2
    ]) - d1

    nB = array([d1 + a2,
                d1 + a2 + a3,
                d3 + a2,
                d3 + a1 + a2
                ]) - d2

    nC = array([d1 + a3,
                d1 + a2 + a3,
                d2 - a2,
                d2 + a3
                ]) - d3

    g = lambda arr: [arctan2(arr[i, 1], arr[i, 0]) for i in range(len(arr))]
    f = lambda arr: [sqrt(arr[i, 0] ** 2 + arr[i, 1] ** 2) for i in range(len(arr))]

    angs = [g(nA), g(nB), g(nC)]

    bls = [f(nA), f(nB), f(nC)]

    tvals = [[t1, t1, t1, t1],
             [t1, t1, t1, t1],
             [t1, t1, t1, t1]]

    angsA = angs[0]
    angsB = angs[1]
    angsC = angs[2]
    si = argsort(angsA)
    print si
    n0 = nA[[0, 3, 2, 1]]
    n1 = nC[[1, 2, 3, 0]]
    n2 = nB[[0, 2, 3, 1]]
    n3 = nA[[3, 0, 1, 2]]
    n4 = nB[[0, 3, 2, 1]]
    n5 = nC[[1, 2, 3, 0]]

    angsG = [[angsA[0], angsA[3], angsA[2], angsA[1]],
             [angsC[1], angsC[2], angsC[3], angsC[0]],
             [angsB[0], angsB[2], angsB[3], angsB[1]],
             [angsA[3], angsA[0], angsA[1], angsA[2]],
             [angsB[0], angsB[3], angsB[2], angsB[1]],
             [angsC[1], angsC[2], angsC[3], angsC[0]]
             ]

    blG = [f(n0), f(n1), f(n2), f(n3), f(n4), f(n5)]
    angsG = [g(n0), g(n1), g(n2), g(n3), g(n4), g(n5)]

    angs = [angsG[i % 6] for i in range(ny)]
    bls = [blG[i % 6] for i in range(ny)]

    tvals = list(t1 * ones_like(angs))

    num_neighbors = zeros((ny, ny), dtype=int)
    for i in range(ny):
        if (i != 0 and i < ny - 2):
            if i % 6 == 0:
                num_neighbors[i, i - 2] = 1
                num_neighbors[i, i - 1] = 1
                num_neighbors[i, i + 1] = 1
                num_neighbors[i, i + 2] = 1

            elif i % 6 == 1:

                num_neighbors[i, i - 1] = 1
                num_neighbors[i, i + 1] = 2
                num_neighbors[i, i + 2] = 1
            elif i % 6 == 2:
                num_neighbors[i, i - 2] = 1
                num_neighbors[i, i - 1] = 2
                num_neighbors[i, i + 1] = 1

            elif i % 6 == 3:
                num_neighbors[i, i - 2] = 1
                num_neighbors[i, i - 1] = 1
                num_neighbors[i, i + 1] = 1
                num_neighbors[i, i + 2] = 1
            elif i % 6 == 4:

                num_neighbors[i, i - 1] = 1
                num_neighbors[i, i + 1] = 2
                num_neighbors[i, i + 2] = 1
            elif i % 6 == 5:
                num_neighbors[i, i - 2] = 1
                num_neighbors[i, i - 1] = 2
                num_neighbors[i, i + 1] = 1

        else:
            if i == 0:
                num_neighbors[i, i + 1] = 1
                num_neighbors[i, i + 2] = 1
            elif i == ny - 1:  # last
                if i % 6 == 0:
                    num_neighbors[i, i - 2] = 1
                    num_neighbors[i, i - 1] = 1


                elif i % 6 == 1:

                    num_neighbors[i, i - 1] = 1

                elif i % 6 == 2:
                    num_neighbors[i, i - 2] = 1
                    num_neighbors[i, i - 1] = 2


                elif i % 6 == 3:
                    num_neighbors[i, i - 2] = 1
                    num_neighbors[i, i - 1] = 1

                elif i % 6 == 4:

                    num_neighbors[i, i - 1] = 1

                elif i % 6 == 5:
                    num_neighbors[i, i - 2] = 1
                    num_neighbors[i, i - 1] = 2
            else:  # (next to last)
                if i % 6 == 0:
                    num_neighbors[i, i - 2] = 1
                    num_neighbors[i, i - 1] = 1
                    num_neighbors[i, i + 1] = 1


                elif i % 6 == 1:

                    num_neighbors[i, i - 1] = 1
                    num_neighbors[i, i + 1] = 2

                elif i % 6 == 2:
                    num_neighbors[i, i - 2] = 1
                    num_neighbors[i, i - 1] = 2
                    num_neighbors[i, i + 1] = 1

                elif i % 6 == 3:
                    num_neighbors[i, i - 2] = 1
                    num_neighbors[i, i - 1] = 1
                    num_neighbors[i, i + 1] = 1

                elif i % 6 == 4:

                    num_neighbors[i, i - 1] = 1
                    num_neighbors[i, i + 1] = 2

                elif i % 6 == 5:
                    num_neighbors[i, i - 2] = 1
                    num_neighbors[i, i - 1] = 2

    b1, b2 = bz.find_bz_lattice_vecs(a3, a1)
    vtx, vty = bz.find_bz_zone(b1, b2)
    vertex_points = array([vtx, vty]).T

    mM = calc_matrix(angs, num_neighbors, bls=bls, tvals=tvals, ons=ons)

    fn = 'kagome_lattice_strip_new_lv'

    fn += '_%0.2f__' % t1234[0]

    fn += '%0.2f_' % ons[0]
    fn += '%0.2f_' % ons[1]
    fn += '%0.2f_' % ons[2]

    fn += '%0.2f_' % (alpha * 180. / pi)

    od = base_dir + '/kagome_lattice_strip_new_lv/'
    if not os.path.isdir(od):
        os.mkdir(od)
        os.mkdir(od + 'data/')
        os.mkdir(od + 'images/')
    return mM, vertex_points, od, fn, [1, 3]
