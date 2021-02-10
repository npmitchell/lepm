import cPickle as pickle
import matplotlib.pyplot as plt
import os
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import sys
import time
from matplotlib import cm
from matplotlib.path import Path
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy import *
from scipy.integrate import dblquad
from scipy.interpolate import griddata


def vec(ang, bl=1):
    # calculates a vector given a bond length and angle in radians
    return [bl * cos(ang), bl * sin(ang)]


def ang_fac(ang):
    return exp(2 * 1j * ang)


def calc_matrix(angs, num_nei, bls=-1, tvals=-1, ons=1, magnetic=False, a=0.85):
    '''Function that returns lambda function of kx and ky for the matrix of a specific lattice
    
    Parameters
    ------------
    angs : list
        each row represents a site in the lattice.  Each entry in the row represents the angles to that site's neighbors
            
    num_nei : list or array (num_sites x num_sites)
        Tells how many neighbors of on each kind of sublattice.  For example a honeycomb lattice would be num_nei = [[0,3], [3,0]] because each point has 3 neighbors of the other lattice type.
        
    bls : array
        dimension equal to dimension of angs.  default value is -1 indicating that all bond lengths are 1
        
    tvals : array
        dimension equal to number of different kinds of springs in unit cell x 1.  represents omega_k
        
    ons : array (dimension = num_sites pre unit cell)
        represents omega_g
        
    Returns
    --------------
    lambda k : make_M(k,angs, num_nei, pin, delta, bls, tvals, ons)
       function for matrix of lattice
    '''

    if magnetic:
        return lambda k: make_M_magnetic(k, angs, num_nei, bls, tvals, ons, a)

    else:

        return lambda k: make_M(k, angs, num_nei, bls, tvals, ons)


def make_M(k, angs, num_neis, bls, tvals, ons):
    """Create a Hamiltonian matrix to diagonalize describing the energetics of a gyro + spring system

    Parameters
    ----------

    Returns
    -------
    """
    ons = list(ons)

    num_sites = len(angs)
    M = zeros([2 * num_sites, 2 * num_sites], dtype='complex128')

    if bls == -1:
        bls = ones_like(angs)
    if tvals == -1:
        tvals = ones_like(angs)
    if ons == 1:
        ons = ones(num_sites)

    for i in range(len(M)):
        index = i % (num_sites)
        angs_for_row = angs[index]
        bls_for_row = bls[index]
        num_neis_row = num_neis[index]
        num_bonds = len(angs[index])

        tv = tvals[index]
        num_bonds = sum(tv)
        # print 'num bonds', num_bonds



        fill_count = 0
        s_fill_count = 0
        for j in range(len(M)):
            if i == j:
                if i < num_sites:

                    M[i, j] = num_bonds + 2 * ons[index]

                else:
                    M[i, j] = - num_bonds - 2 * ons[index]
            else:
                ii = j % num_sites
                num_nei = num_neis_row[ii]

                if i < num_sites and j < num_sites:
                    for l in range(num_nei):
                        M[i, j] += -tv[fill_count] * exp(
                            1j * dot(k, vec(angs_for_row[fill_count], bls_for_row[fill_count])))
                        fill_count += 1
                elif i >= num_sites and j >= num_sites:
                    for l in range(num_nei):
                        M[i, j] += tv[fill_count] * exp(
                            1j * dot(k, vec(angs_for_row[fill_count], bls_for_row[fill_count])))
                        fill_count += 1
                elif i < num_sites and j >= num_sites:
                    if j == num_sites + i:

                        M[i, j] = sum([tv[u] * ang_fac(angs_for_row[u]) for u in range(len(angs_for_row))])
                    else:

                        for l in range(num_nei):
                            ss = time.time()
                            M[i, j] += -tv[s_fill_count] * ang_fac(angs_for_row[s_fill_count]) * exp(
                                1j * dot(k, vec(angs_for_row[s_fill_count], bls_for_row[s_fill_count])))
                            sd = time.time()

                            s_fill_count += 1

                elif i >= num_sites and j < num_sites:
                    if j == (num_sites + i) % num_sites:
                        M[i, j] = -sum([tv[u] * ang_fac(-angs_for_row[u]) for u in range(len(angs_for_row))])
                    else:

                        for l in range(num_nei):
                            M[i, j] += tv[s_fill_count] * ang_fac(-angs_for_row[s_fill_count]) * exp(
                                1j * dot(k, vec(angs_for_row[s_fill_count], bls_for_row[s_fill_count])))
                            s_fill_count += 1

    if False:  # m_or_g != 'gyro':
        M[num_sites:] = -M[num_sites:]

    return -0.5 * M


def make_M_magnetic(k, angs, num_neis, bls, tvals, ons, a=0.9):
    a = 0.8
    # print 'a is', a

    # print 'delta is ', delta
    num_sites = len(angs)
    Ok = tvals[0][0]

    tvals = list(ones_like(tvals))
    if a > 0:

        a = a ** 2.
        Op = (1. + a / 6. - (1. / 4 + a / 12.)) * Ok
        Om = (1. + a / 6. + (1. / 4 + a / 12.)) * Ok
        # pin =  1 - 3./8*a*Ok
        # print Op
        # print Om
    else:
        Op = 1.
        Om = 1.

    M = zeros([2 * num_sites, 2 * num_sites], dtype='complex')

    if bls == -1:
        bls = ones_like(angs)
    if tvals == -1:
        tvals = ones_like(angs)
    if ons == 1:
        ons = ones(num_sites)

    for i in range(len(M)):
        index = i % (num_sites)
        angs_for_row = angs[index]
        bls_for_row = bls[index]
        num_neis_row = num_neis[index]
        num_bonds = len(angs[index])

        tv = tvals[index]
        num_bonds = sum(tv)

        ff = 0
        tt = tv[2] * (1 + 2 * sin(20 * pi / 180))

        fill_count = 0
        s_fill_count = 0
        for j in range(len(M)):
            if i == j:
                if i < num_sites:

                    M[i, j] = Op * (num_bonds) + 2 * (ons[index] - (3. / 8) * a * Ok)

                else:
                    M[i, j] = - Om * (num_bonds) - 2 * (ons[index] - (3. / 8) * a * Ok)
            else:
                ii = j % num_sites
                num_nei = num_neis_row[ii]
                # print 'num nei',  num_nei

                if i < num_sites and j < num_sites:
                    for l in range(num_nei):
                        M[i, j] += -Op * tv[fill_count] * exp(
                            1j * dot(k, vec(angs_for_row[fill_count], bls_for_row[fill_count])))
                        fill_count += 1
                elif i >= num_sites and j >= num_sites:
                    for l in range(num_nei):
                        M[i, j] += Om * tv[fill_count] * exp(
                            1j * dot(k, vec(angs_for_row[fill_count], bls_for_row[fill_count])))
                        fill_count += 1
                elif i < num_sites and j >= num_sites:
                    if j == num_sites + i:

                        M[i, j] = Om * sum([tv[u] * ang_fac(angs_for_row[u]) for u in range(len(angs_for_row))])
                        # last_fac = angs_for_row[2] +pi
                        # M[i,j] += ff*Om*tt*ang_fac(last_fac)
                    else:

                        for l in range(num_nei):
                            ss = time.time()
                            M[i, j] += -Om * tv[s_fill_count] * ang_fac(angs_for_row[s_fill_count]) * exp(
                                1j * dot(k, vec(angs_for_row[s_fill_count], bls_for_row[s_fill_count])))
                            sd = time.time()

                            s_fill_count += 1

                elif i >= num_sites and j < num_sites:
                    if j == (num_sites + i) % num_sites:
                        M[i, j] = -Op * sum([tv[u] * ang_fac(-angs_for_row[u]) for u in range(len(angs_for_row))])
                    else:

                        for l in range(num_nei):
                            M[i, j] += Om * tv[s_fill_count] * ang_fac(-angs_for_row[s_fill_count]) * exp(
                                1j * dot(k, vec(angs_for_row[s_fill_count], bls_for_row[s_fill_count])))
                            s_fill_count += 1

    return -0.5 * M


def outer_hexagon(size): return array([(size * sin(i * pi / 3), size * cos(i * pi / 3)) for i in range(6)]), 'hexagon'


def calc_eigvects_eigvals(g, kx, ky):
    """

    Parameters
    ----------
    g
    kx
    ky

    Returns
    -------

    """
    M = g([kx, ky])
    eigval, eigvect = linalg.eig(M)
    eigval = real(eigval)

    eigvect = eigvect.T
    si = argsort(eigval)

    eigval = eigval[si]
    eigvect = eigvect[si]

    return eigval, eigvect


def calc_p(eigvect):
    """

    Parameters
    ----------
    eigvect

    Returns
    -------

    """
    p = []
    le = len(eigvect)
    for j in range(le):
        p.append(conjugate(reshape((eigvect)[j], [le, 1])) * reshape(eigvect[j], [1, le]))

    return [p, eigvect]


def func(g): return lambda k: make_f(g, k)


def make_f(g, k):
    kx = k[0]
    ky = k[1]

    eigval, eigvect = calc_eigvects_eigvals(g, kx, ky)
    p_v, eigvect = calc_p(eigvect)
    obj = [array(p_v), eigval, eigvect]

    return obj


def calc_numerical(func, h, k):
    """Calculates a 2D numerical derivative
    
    Parameters
    ------------
    func : function you want derivative of
            
    h : float
        spacing in BZ for derivative calculation.
        
    K : 2x1 float array
        points at kx, ky in the BZ
      
    Returns
    -------
    [fpx, fpy]:
        derivatives along x and y 
    """
    x = k[0]
    y = k[1]

    # ss = time.time()
    # sd = time.time()

    ax = func([x + 2 * h, y])[0]
    bx = func([x + h, y])[0]
    cx = func([x - h, y])[0]
    dx = func([x - 2 * h, y])[0]

    ay = func([x, y + 2 * h])[0]
    by = func([x, y + h])[0]
    cy = func([x, y - h])[0]
    dy = func([x, y - 2 * h])[0]

    fpx = (-ax + 8 * bx - 8 * cx + dx) / (12 * h)
    fpy = (-ay + 8 * by - 8 * cy + dy) / (12 * h)

    return [fpx, fpy]


def generate_kx_ky_random(NP, vertex_points, crx=6, cry=6, len_prev=0, **kwargs):
    NP_new = NP - len_prev

    if 'center' in kwargs:
        sl = vertex_points[0, 0] - vertex_points[1, 0]
        print 'sl is', sl
        center = kwargs['center']
        kx = (sl) * (random.rand(NP_new) - 0.5) + center[0]
        ky = (sl) * (random.rand(NP_new) - 0.5) + center[1]




    else:
        kx = (2 * crx) * random.rand(NP_new) - crx
        ky = (2 * cry) * random.rand(NP_new) - cry

    poly_path = Path(vertex_points)
    i_vals = array([i for i in range(NP_new) if poly_path.contains_point([kx[i], ky[i]])])

    kx = kx[i_vals]
    ky = ky[i_vals]

    while len(kx) < (NP - len_prev):
        NP_new = NP - len(kx) - len_prev + 20
        kkx = (2 * crx) * random.rand(NP_new) - crx
        kky = (2 * cry) * random.rand(NP_new) - cry
        i_vals = array([i for i in range(len(kkx)) if poly_path.contains_point([kkx[i], kky[i]])])

        kkx = kkx[i_vals]
        kky = kky[i_vals]

        kx = concatenate((kx, kkx))
        ky = concatenate((ky, kky))

    return kx, ky


def generate_kx_ky_grid(delta_grid, vertex_points, crx=4, cry=4):
    """Generate a regular square grid of kx, ky values that lie within vertex_points,
    within bounds (-crx, crx), (-cry, cry)

    Parameters
    ----------
    delta_grid
    vertex_points
    crx
    cry

    Returns
    -------

    """
    x = np.arange(-crx, crx, delta_grid)
    y = np.arange(-cry, cry, delta_grid)
    kx, ky = np.meshgrid(x, y)
    kx = kx.flatten()
    ky = ky.flatten()
    NP = len(kx)

    poly_path = Path(vertex_points)
    i_vals = array([i for i in range(NP) if poly_path.contains_point([kx[i], ky[i]])])

    return kx, ky


def calc_bands(mM, Kx, Ky):
    """calculates the eigenvalues and traces for some point kx, ky in the Bz
    
    Parameters
    ------------
    mM : lambda function
        each row represents a site in the lattice.  Each entry in the row represents the angles to that site's neighbors
    Kx : float
        Tells how many neighbors of on each kind of sublattice.  For example a honeycomb lattice would be num_nei = [[0,3], [3,0]] because each point has 3 neighbors of the other lattice type.
    Ky : float
        dimension equal to dimension of angs.  default value is -1 indicating that all bond lengths are 1
      
    Returns
    -------
    eigval : float array length 2*num_sites
         floats for band structure at input values of Kx and Ky
    tr : float array length 2*num_sites
         Berry curvature for bands in the BZ zone at kx, ky
    """
    h = 10 ** -5
    # func is a lambda function defined above. (what a great name)
    der_func = func(mM)
    # func (and therefore der_func) is a lambda function.
    p_v, eigval, eigvect = der_func([Kx, Ky])
    # I think I just did this so that I would have one function of kx and ky to take the derivative of
    # in the calc_numerical.

    # calc_numerical is short for 'calculate numerical derivative
    fpx, fpy = calc_numerical(der_func, h, [Kx, Ky])
    # note: you could probably use a canned function to calculate the numerical derivative, but I didn't.

    tr = []
    for j in range(len(p_v)):
        t1 = dot(fpx[j], dot(p_v[j], fpy[j]))
        t2 = dot(fpy[j], dot(p_v[j], fpx[j]))
        tr.append(trace(t1 - t2))

    # tr is what you integrate over the BZ to get the chern number.
    return eigval, tr


def save_pickled_data(output_dir, filename, data):
    """saves data in a pickle file.
    
    Parameters
    ----------
    output_dir : string
        Directory in which to save file
    filename : string 
        name of file
    data : any python object
        python object you want to save
    """
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    of = open(output_dir + '/' + filename + '.pickle', 'wb')
    pickle.dump(data, of, pickle.HIGHEST_PROTOCOL)
    print 'saved', output_dir + '/' + filename + '.pickle'
    of.close()


def save_pickled_data_full(output_dir, data):
    """saves data in a pickle file.
    
    Parameters
    ----------
    output_dir : string
        Directory in which to save file
    filename : string 
        name of file
    data : any python object
        python object you want to save
    """
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    of = open(output_dir, 'wb')
    pickle.dump(data, of, pickle.HIGHEST_PROTOCOL)
    print 'saved ', output_dir
    of.close()


def PolygonArea(corners):
    """

    Parameters
    ----------
    corners

    Returns
    -------

    """
    n = len(corners)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area


def find_nearest(array, value):
    """Find the indices of the 2d array 'array' that are closest to the 2d pt 'value'"""
    xvals = array[:, 0]
    yvals = array[:, 1]

    vx = value[0]
    vy = value[1]

    diffx = abs(xvals - vx)
    diffy = abs(yvals - vy)

    idx = (abs(diffx ** 2 + diffy ** 2)).argmin()
    return idx
