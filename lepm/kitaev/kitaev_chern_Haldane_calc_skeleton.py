import numpy as np
import numpy.linalg as la
import lepm.lattice_elasticity as le
from lepm import lattice_class
import lepm.haldane.haldane_lattice_functions as hlatfns
import matplotlib.pyplot as plt
import lepm.build.make_lattice as makeL
import copy

'''Calculate Chern number via realspace method Kitaev method.
This code shows the calculation in the fewest low-level function calls possible.
'''

# Parameters
NH = 30
NV = 30
Omg = -10.
Omk = -1.0
LT = 'hucentroid'
shape = 'square'
delta = np.pi*120./180.
periodicBC = False
epsilon = Omk**2 / Omg
ksize = 14.01933796281622335

# Make Lattice
lp = {'LatticeTop': LT, 'shape': shape, 'NH': NH, 'NV': NV, 'delta': delta, 'phi': 0., 'eta': 0., 'theta': 0.,
      'periodicBC': periodicBC, 'check': False,
      'rootdir': '/Users/npmitchell/Dropbox/Soft_Matter/GPU/', 'NP_load': 0, 'conf': 01,
      'origin': np.array([0, 0]), 'Omg': Omg, 'Omk': Omk}

try:
    lat = lattice_class.Lattice(lp=lp)
    lat.load()
    xy = lat.xy
except:
    print 'Could not load lattice, so creating it from scratch'
    if lp['LatticeTop'] == 'hexagonal':
        xy, NL, KL, BL, LVUC, LV, UC, PVxydict, PVx, PVy, lattice_exten = makeL.generate_honeycomb_lattice(lp)
    else:
        xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten = makeL.build_hucentroid(lp)

    lat = lattice_class.Lattice(lp=lp, xy=xy, NL=NL, KL=KL, BL=BL, PVxydict=PVxydict, PVx=PVx, PVy=PVy)

# Make Dynamical Matrix
# eigvect, eigval, DM = hlatfns.normal_modes_haldane(xy, NL, KL, BL, datadir=None, epsilon=epsilon,
#                                                    save_ims=False, PVx=PVx, PVy=PVy)
eigvect, eigval, DM = hlatfns.normal_modes_haldane(lat, epsilon=epsilon)

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
h = np.zeros((len(P), len(P), len(P)), dtype=complex)
# for j in range(len(P)):
#     for k in range(len(P)):
#         for l in range(len(P)):
#             h[j,k,l] = P[j,k]*P[k,l]*P[l,j] - P[j,l]*P[l,k]*P[k,j]

# h *= 12*np.pi*1j

# Divide plane into three, A->B->C counterclockwise
theta1 = -np.pi/6.
theta2 = np.pi*(7./6.)
teps = 1e-6  # small offset of region boundaries
polygon1 = ksize*np.array([[0., 0.], [np.cos(theta1), np.sin(theta1)], [0., 1.]])
polygon2 = ksize*np.array([[0., 0.], [0., 1.], [np.cos(theta2), np.sin(theta2)]])
polygon3 = ksize*np.array([[0., 0.], [np.cos(theta2+teps), np.sin(theta2+teps)],
                           [np.cos(theta1-teps), np.sin(theta1-teps)]])
reg1 = le.inds_in_polygon(xy, polygon1)
reg2 = le.inds_in_polygon(xy, polygon2)
reg3 = le.inds_in_polygon(xy, polygon3)

print 'Summing up h values...'
dmyi = 0
nu = 0. + 0.*1j
for j in reg1:
    for k in reg2:
        for l in reg3:
            # print 'jkl = ', j, ',', k, ',', l
            # nu += h[j,k,l]
            nu += 12*np.pi*1j*(P[j, k]*P[k, l]*P[l, j] - P[j, l]*P[l, k]*P[k, j])
            dmyi += 1

print 'nu = ', nu

