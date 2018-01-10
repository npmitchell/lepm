import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import lattice_elasticity as le
import plotting.colormaps as colormaps

phi = 0.0 * np.pi
delta = np.pi * 1.1 #2./3.
theta = 0.5 * (np.pi - delta)

pts = np.array([[np.sin(phi), np.cos(phi)],
                [np.sin(phi) + np.cos(theta), np.cos(phi) + np.sin(theta)],
                [np.sin(phi) + 2*np.cos(theta), np.cos(phi)],
                [2*np.cos(theta), 0.],
                [np.cos(theta), -np.sin(theta)],
                [0., 0.]
                ])

colormaps.register_colormaps()
xy = np.vstack((pts, pts[1:5]+np.array([2*np.cos(theta),0]), pts[0:3] + pts[1], pts[1:4] + pts[1]+pts[3]))
BL = np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[0,5],
               [2,6],[6,7],[7,8],[8,9],[3,9],
               [1,10],[10,11],[11,12],[6,12],
               [12,13],[13,14],[14,15],[7,15]])
NL, KL = le.BL2NLandKL(BL)
NLNNN, KLNNN = le.calc_NLNNN_and_KLNNN(xy,BL,NL,KL,PVx=None,PVy=None)
print 'NLNNN = ', NLNNN
print 'KLNNN = ', KLNNN
filename = '/Users/npmitchell/Desktop/visualizeNNN_phi'+'{0:0.3f}'.format(phi) + '{0:03f}'.format(delta)+'.png'
le.movie_plot_2D(xy, BL, 0*BL[:,0], filename, '', NL=NL, KL=KL, NLNNN=NLNNN, KLNNN=KLNNN, PVxydict={},
                      xlimv=None, ylimv=None, climv=0.1, lw=2., colorz=False, axcb='none', axis_off=True, \
                      colorpoly=False, show=True, colormap='BlueBlackRed', bgcolor = '#ffffff', arrow_alpha=0.3)


# Sides as clockwise vectors
s0 = pts[1] - pts[0]
s1 = pts[2] - pts[1]
s2 = pts[3] - pts[2]
s3 = pts[4] - pts[3]
s4 = pts[5] - pts[4]
s5 = pts[0] - pts[5]
print 'sides = ', s0, s1, s2, s3, s4, s5

# Theta angles of each side vector
t0 = np.arctan2(s0[1], s0[0])% (2.*np.pi)
t1 = np.arctan2(s1[1], s1[0])% (2.*np.pi)
t2 = np.arctan2(s2[1], s2[0])% (2.*np.pi)
t3 = np.arctan2(s3[1], s3[0])% (2.*np.pi)
t4 = np.arctan2(s4[1], s4[0])% (2.*np.pi)
t5 = np.arctan2(s5[1], s5[0])% (2.*np.pi)
print 'angles = ', t0, t1, t2, t3, t4, t5

# The fluxes for the a sites
a0 = (t0 - t1) % (2.*np.pi)
a1 = (t2 - t3) % (2.*np.pi)
a2 = (t4 - t5) % (2.*np.pi)
a3 = -a0 #% (2.*np.pi)
a4 = -a1 #% (2.*np.pi)
a5 = -a2 #% (2.*np.pi)
print 'a fluxes = ', a0, a1, a2, a3, a4, a5

# The fluxes for the b sites
b0 = (t3 - t4) % (2.*np.pi)
b1 = (t1 - t2) % (2.*np.pi)
b2 = (t5 - t0) % (2.*np.pi)
b3 = -b0 #% (2.*np.pi)
b4 = -b1 #% (2.*np.pi)
b5 = -b2 #% (2.*np.pi)
print 'b fluxes = ', b0, b1, b2, b3, b4, b5

# Solve linear system of equations for fluxes
c = sp.Symbol('c')
d = sp.Symbol('d')
e = sp.Symbol('e')
f = sp.Symbol('f')
g = sp.Symbol('g')
h = sp.Symbol('h')
i = sp.Symbol('i')
j = sp.Symbol('j')
k = sp.Symbol('k')
l = sp.Symbol('l')
m = sp.Symbol('m')
n = sp.Symbol('n')
p = sp.Symbol('p')
result = sp.solve([a0-c-d-e, a1-g-h-i, a2-l-m-k, b3-i-j-k, b4-e-f-g, b5-m-n-c,
                   -a0-a1-a2-f-j-n-p, b0+b1+b2-d-h-l-p,
                   a0+a1+a2-c-d-e-g-h-i-m-l-k,                         # from outer triangle 1
                   b3+b4+b5-i-j-k-e-f-g-m-n-c,                         # from outer triangle 2
                   c + d + e + f + g + h + i + j + k + l + m + n + p,  # from parallelogram b0+b5+b3+b2 = 0
                   c, d, e, i, j, k,
                   ],
                  [c, d, e, f, g, h, i, j, k, l, m, n, p])
# Check that first outer triangle is a0 a1 a2
# Check that pentagon is distinct: -a1-a1-a0+a1-a2 = p+n+f+j+c+d+e+k+j+i+p+f+m+n+g+h
# A very large triangle: b0+b0+b1+b1+b2+b2
#c+d+e+f+g+h+i+j+k+l+m+n+p
#, d, e, f, g, h, i, j, k, l, m, n, p

print 'result: ', result
