import numpy as np
import argparse
import numpy.linalg as la
import lepm.lattice_elasticity as le
import matplotlib.pyplot as plt
from os import getcwd,chdir
import glob
import subprocess
import copy
import cPickle
from scipy.optimize import curve_fit
import argparse
import sys

'''Calculate Chern number via realspace method, as a script.

inverse --> transpose dagger (conjugate transp)
U Udagger should also give identity, if orthonormal

Check
--> orthogonality
--> normalization
so that evects is orthonormal
so that U Udagger = I


The eigenvalues for the left and right matrices are identical (A and A.T)
'''

parser = argparse.ArgumentParser('Evolve a spring+gyro system on GPU')
parser.add_argument('lattice', type=str, nargs='?',
                    help='Kind of lattice simulation to open',
                    default='check_string_for_empty')
parser.add_argument('-size', '--size', help='Size of lattice (small, medium, large)', type=str, default='medium')
parser.add_argument('-Levin_P', '--Levin_P', help='Whether to use M. Levins method to get P', action='store_true')
parser.add_argument('-XY_basis', '--XY_basis', help='Reclare to use XY basis instead of psi_L,R', action='store_true')
args = parser.parse_args()

# What lattice to look for
rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/'
if args.lattice == 'honeycomb':
    if args.size == 'small':
        seriesdir = 'NashCL_hexagonal_3particle/20160126-0923_h0p01_Omk1p000_Omg1p000'
    if args.size == 'medium':
        seriesdir = 'NashCL_hexagonal_6particle/20160226-1535_NV6_Omk1p000_Omg1p000'
    # Large
    # seriesdir = ''
elif args.lattice == 'bowtie':
    if args.size == 'medium':
        seriesdir = 'NashCL_hexagonal_6particle_disordered_bowtie_delta1p25/20160207-1108_NV6_Omk1p000_Omg1p000' 
elif args.lattice == 'linear':
    seriesdir = 'NashCL_linear_2particle/20160304-1806_NV2_Omk1p000_Omg1p000'

datadir = rootdir+'experiments/'+seriesdir+'/'
if args.XY_basis:
    outdir = datadir+'chern/XY_basis/'
else:
    outdir = datadir+'chern/psi_basis/'
le.ensure_dir(outdir)

# Load eigenvals and eigenvects
if glob.glob(datadir+'eigval_psiT.pkl'):
    with open(datadir+'eigval_psi.pkl', "rb") as input_file:
        eigval = cPickle.load(input_file)
    with open(datadir+'eigvect_psi.pkl', "rb") as input_file:
        eigvect = cPickle.load(input_file)
    with open(datadir+'eigval.pkl', "rb") as input_file:
        eigvalXY = cPickle.load(input_file)
    with open(datadir+'eigvect.pkl', "rb") as input_file:
        eigvectXY = cPickle.load(input_file)
    with open(datadir+'eigval_psiT.pkl', "rb") as input_file:
        eigvalT = cPickle.load(input_file)
    with open(datadir+'eigvect_psiT.pkl', "rb") as input_file:
        eigvectT = cPickle.load(input_file)
else:
    print 'Solving psi-rep eigenvalue problem'
    params = le.load_params(datadir)
    xyload = np.loadtxt(datadir+'xy.txt', delimiter=',', skiprows=1, usecols=(0,1), unpack=True)
    KLload = np.loadtxt(datadir+'KL.txt', delimiter=',', skiprows=1)
    NLload = np.loadtxt(datadir+'NL.txt', delimiter=',', skiprows=1)
    Omgload = np.loadtxt(datadir+'Omg.txt', delimiter=',', skiprows=1)
    OmKload = np.loadtxt(datadir+'OmK.txt', delimiter=',', skiprows=1)
    OmK = OmKload.astype(float)
    Omg = Omgload.astype(float)
    xy0 = (xyload.T).astype(float)
    xy = np.dstack((xy0[:,0],xy0[:,1]))[0]
    NL = (NLload).astype(np.intc)
    KL = (KLload).astype(np.intc)
    NP = np.shape(KL)[0]
    NN = np.shape(KL)[1]
    eigvect, eigval, DM = le.save_normal_modes_Nashgyro_psirep(datadir,xy, NL, KL,OmK, Omg, params, sim_type='gyro')
    np.savetxt(datadir+'dynamical_matrix_psi.txt',DM, delimiter=',')
    #le.plot_complex_matrix(DM,name='Dyn. M.',outpath=datadir+'dynamical_matrix',show=False,clear=True)
    
    with open(datadir+'eigval_psi.pkl', "rb") as input_file:
        eigval = cPickle.load(input_file)
    with open(datadir+'eigvect_psi.pkl', "rb") as input_file:
        eigvect = cPickle.load(input_file)
    with open(datadir+'eigval.pkl', "rb") as input_file:
        eigvalXY = cPickle.load(input_file)
    with open(datadir+'eigvect.pkl', "rb") as input_file:
        eigvectXY = cPickle.load(input_file)
    with open(datadir+'eigval_psiT.pkl', "rb") as input_file:
        eigvalT = cPickle.load(input_file)
    with open(datadir+'eigvect_psiT.pkl', "rb") as input_file:
        eigvectT = cPickle.load(input_file)
    
# Define projector operator
# First define U such that A = U D U^{-1},
# where D = diag(eigenvals), and A is dynamical matrix
# Let M be D with zeros where omega>band1 (or omega<band2)
# Then P_ij = U M U^{-1}.
if args.XY_basis:
    eigvect = eigvectXY

U = eigvect.transpose()

# for row in eigvect:
#     print ' norm= ', la.norm(row)
# 
# sys.exit()
# Make sure orthogonal
U1 = la.inv(U)


''' Is U U1 - I =0?'''
UU1 = np.dot(U, U1)
III = np.eye(len(U))
# plt.clf()
# plt.plot(np.arange(len(UU1.ravel())), np.imag(UU1.ravel()-III.ravel()),'b.' )
# plt.plot(np.arange(len(UU1.ravel())), np.real(UU1.ravel()-III.ravel()),'r.' )
# ax = plt.gca()
# ax.set_title(r'$U U^{-1} - I$')
# plt.show()

'''Are eigenvects orthonormal?'''
Umatrix = np.matrix(U)
Udagger = U.conj().T
UUdagger = np.dot(U,Udagger)
plt.clf()
plt.plot(np.arange(len(UU1.ravel())), np.imag(UUdagger.ravel()-III.ravel()),'b.' )
plt.plot(np.arange(len(UU1.ravel())), np.real(UUdagger.ravel()-III.ravel()),'r.' )
ax = plt.gca()
ax.set_title(r'$ U U^{dagger} - I$')
plt.savefig(outdir+'UUdagger.png')
plt.clf()

D = np.zeros((len(U), len(U)), dtype=complex)
for ii in range(len(eigval)):
    if args.XY_basis:
        ev = eigvalXY[ii]
        D[ii, ii] = ev
    else:
        ev = eigval[ii]
        D[ii, ii] = ev


plt.plot(np.arange(len(eigval.ravel())),np.imag(eigval.ravel()),'b.',label='imaginary part')
plt.plot(np.arange(len(eigval.ravel())),np.real(eigval.ravel()),'r.',label='real part')
plt.plot(np.arange(len(eigvalXY.ravel())),np.imag(eigvalXY.ravel()),'g.',label='imaginary part XY')
plt.ylabel(r'$\omega$')
plt.xlabel(r'Index of $eigval$')
ax = plt.gca()
ax.set_title(r'elements of $eigval$')
ax.legend(loc='best')
plt.savefig(outdir+'eigval.png')

print '\nD = ', D
print '\nU = ', U
plt.clf()
plt.plot(np.arange(len(D.ravel())),np.imag(D.ravel()),'b.',label='imaginary part')
plt.plot(np.arange(len(D.ravel())),np.real(D.ravel()),'r.',label='real part')
plt.ylabel(r'$\omega$')
plt.xlabel(r'Index of $D$')
ax = plt.gca()
ax.set_title(r'Elements of $D$')
ax.legend(loc='best')
plt.savefig(outdir+'D.png')

    
    
# Verify that A = U D U^{-1}
params = le.load_params(datadir)
xyload = np.loadtxt(datadir+'xy.txt', delimiter=',', skiprows=1, usecols=(0,1), unpack=True)
KLload = np.loadtxt(datadir+'KL.txt', delimiter=',', skiprows=1)
NLload = np.loadtxt(datadir+'NL.txt', delimiter=',', skiprows=1)
Omgload = np.loadtxt(datadir+'Omg.txt', delimiter=',', skiprows=1)
OmKload = np.loadtxt(datadir+'OmK.txt', delimiter=',', skiprows=1)
OmK = OmKload.astype(float)
Omg = Omgload.astype(float)
xy0 = (xyload.T).astype(float)
xy = np.dstack((xy0[:,0],xy0[:,1]))[0]
NL = (NLload).astype(np.intc)
KL = (KLload).astype(np.intc)
NP = np.shape(KL)[0]
NN = np.shape(KL)[1]
print '\nNP = ', NP
A = le.dynamical_matrix_psi(xy, NL, KL, OmK, Omg, params)
#A = le.normal_modes_gyros(xy, NL, KL, OmK, Omg, params, dispersion=[], sublattice_labels=[])
print '\nA - U D U^{-1} = ',
print '\n\nA - np.dot(U,np.dot(D,U1)) = ', A - np.dot(U,np.dot(D,U1))



if args.Levin_P:
    # Let M be A with zeros where omega>band1 (or omega<band2)
    M = copy.deepcopy(D)
    print 'M = ', M
    print 'np.count_nonzero(M) = ', np.count_nonzero(M)
    plt.plot(np.arange(len(M.ravel())),np.imag(M.ravel()),'b.',label='imaginary part')
    plt.plot(np.arange(len(M.ravel())),np.real(M.ravel()),'r.',label='real part')
    plt.ylabel(r'$\omega$')
    plt.xlabel(r'Index of $M$')
    ax = plt.gca()
    ax.set_title(r'$M$ : copy of $D$')
    ax.legend(loc='best')
    plt.savefig(outdir+'M_start.png')
    #plt.show()
    
    # M[M.imag<0] = 0 #kill negative energies
    # M[M.imag>1.95] = 0 # kill above the lower pos band
    # kill below bottom of top band
    if args.size == 'small':
        if args.XY_basis:
            print 'XY basis: using this XY'
            omegac = 2.5
        else:
            omegac = 4.15 
        M[M.imag < omegac] = 0
        M[M.imag > omegac] = 1
    elif args.size == 'medium':
        if args.lattice == 'honeycomb':
            omegac = 4.15
            # M[M.imag<2.551] = 0
            # M[M.imag>2.551] = 1
            M[M.imag < omegac] = 0
            M[M.imag > omegac] = 1
        elif args.lattice == 'bowtie':
            M[M.imag < 2.405] = 0
            M[M.imag > 2.405] = 1
    print 'M = ', M
    
    print 'np.count_nonzero(M) = ', np.count_nonzero(M)
    plt.clf()
    plt.plot(np.arange(len(M.ravel())),np.imag(D.ravel()),'g.',label='imaginary part of D (for comparison)')
    plt.plot(np.arange(len(M.ravel())),np.imag(M.ravel()),'b.',label='imaginary part')
    plt.plot(np.arange(len(M.ravel())),np.real(M.ravel()),'r.',label='real part')
    plt.plot()
    plt.ylabel(r'$\omega$')
    plt.xlabel(r'Index of $M$')
    ax = plt.gca()
    ax.set_title(r'$M$ : filtered $D$ as ones')
    ax.legend(loc='best')
    plt.savefig(outdir+'M_filtered.png')
    plt.clf()
    #plt.show()
    
    fig = plt.figure()
    a=fig.add_subplot(1,2,1)
    img = a.imshow(np.imag(M),cmap="coolwarm")
    a.set_title('Imaginary part of $M_{Levin}$')
    plt.colorbar(img, orientation ='horizontal')
    a=fig.add_subplot(1,2,2)
    img2 = a.imshow(np.real(M),cmap="coolwarm")
    a.set_title('Real part of $M_{Levin}$')
    plt.colorbar(img2,orientation='horizontal')
    plt.savefig(outdir+'M_visual.png')
    plt.clf()
    
    # Then P_ij = U M U^{-1}.
    P = np.dot(U, np.dot(M,U1))
    #P[NP:] *= -1
    P2 = np.dot(P,P)
    print 'P = ', P
    print 'P^2 = ', P2
    print 'max(P.imag) = ', np.max(P.imag.ravel())
    print 'max(P.real) = ', np.max(P.real.ravel())
    print 'shape(P) = ', np.shape(P)
    print 'len(xy) = ', len(xy)
    plt.clf()
    plt.plot(np.arange(len(P.ravel())),np.imag(P.ravel()),'b.',label='imaginary part')
    plt.plot(np.arange(len(P.ravel())),np.real(P.ravel()),'r.',label='real part')
    plt.ylabel(r'$P_{ij}$')
    plt.xlabel(r'Index of $P$')
    ax = plt.gca()
    ax.set_title(r'$P$')
    ax.legend(loc='best')
    plt.savefig(outdir+'P.png')
    plt.clf()
    
    fig = plt.figure()
    a=fig.add_subplot(1,2,1)
    img = a.imshow(np.imag(P),cmap="coolwarm")
    a.set_title('Imaginary part of $P_{Levin}$')
    plt.colorbar(img, orientation ='horizontal')
    a=fig.add_subplot(1,2,2)
    img2 = a.imshow(np.real(P),cmap="coolwarm")
    a.set_title('Real part of $P_{Levin}$')
    plt.colorbar(img2,orientation='horizontal')
    plt.savefig(outdir+'P_visual.png')
    plt.clf()
    
    plt.clf()
    plt.plot(np.arange(len(P.ravel())),np.imag(P2.ravel()),'b.',label='imaginary part')
    plt.plot(np.arange(len(P.ravel())),np.real(P2.ravel()),'r.',label='real part')
    plt.ylabel(r'$P_{ij}^2$')
    plt.xlabel(r'Index of $P$')
    ax = plt.gca()
    ax.set_title(r'$P^2$')
    ax.legend(loc='best')
    plt.savefig(outdir+'P2.png')
    plt.clf()
    
    P2 = np.dot(P,P) #.conj().transpose())
    fig = plt.figure()
    a=fig.add_subplot(1,2,1)
    img = a.imshow(np.imag(P2),cmap="coolwarm")
    a.set_title('Imaginary part of $P^2_{Levin}$')
    plt.colorbar(img, orientation ='horizontal')
    a=fig.add_subplot(1,2,2)
    img2 = a.imshow(np.real(P2),cmap="coolwarm")
    a.set_title('Real part of $P^2_{Levin}$')
    plt.colorbar(img2,orientation='horizontal')
    plt.savefig(outdir+'P2_visual.png')
    plt.clf()
    
    plt.clf()
    plt.plot(np.arange(len(P.ravel())),np.imag(P2.ravel())-np.imag(P.ravel()),'b.',label='imaginary part')
    plt.plot(np.arange(len(P.ravel())),np.real(P2.ravel())-np.real(P.ravel()),'r.',label='real part')
    plt.ylabel(r'$P_{ij}^2$')
    plt.xlabel(r'Index of $P$')
    ax = plt.gca()
    ax.set_title(r'$P^2 - P$')
    ax.legend(loc='best')
    plt.savefig(outdir+'P2_minus_P.png')
    #plt.show()
else:
    '''Are left and right eigenvals the same?'''
    eL = eigval
    eR = eigvalT
    plt.plot(np.arange(len(eL)), np.imag(eL),'bo')
    plt.plot(np.arange(len(eR)), np.imag(eR),'r.')
    # plt.show()
    plt.savefig(outdir+'right_and_left_eigenvects_psi.png')
    plt.clf()
    
    '''Make V, which is the eigenvects of AT'''
    V = eigvectT.transpose()
    fig = plt.figure()
    a=fig.add_subplot(1, 2, 1)
    img = a.imshow(np.imag(V), cmap="coolwarm")
    a.set_title('Imaginary part of eigenvects of $A^T$')
    plt.colorbar(img, orientation='horizontal')
    a=fig.add_subplot(1, 2, 2)
    img2 = a.imshow(np.real(V), cmap="coolwarm")
    a.set_title('Real part of eigenvects of $A^T$')
    plt.colorbar(img2, orientation='horizontal')
    plt.savefig(outdir + 'V_eigvects_AT.png')
    plt.clf()

    omegac = 4.25
    nplus = np.where( np.imag(eigval) > omegac)[0]
    nminus = np.setdiff1d(np.arange(len(nplus)), nplus)
    P = np.zeros((len(eigval),len(eigval)))
    for i in nplus:
        P += np.outer(U[:,i], V[:,i]) + np.outer(V[:,i], U[:,i])
    for i in nminus:
        P -= np.outer(U[:,i], V[:,i]) + np.outer(V[:,i], U[:,i])
    
    print 'nplus = ', nplus
    print 'nminus = ', nminus
    print 'P = ', P
    
    # Visualize P
    fig = plt.figure()
    a=fig.add_subplot(1,2,1)
    img = a.imshow(np.imag(P),cmap="coolwarm")
    a.set_title('Imaginary part of $P_{suri}$')
    plt.colorbar(img, orientation ='horizontal')
    a=fig.add_subplot(1,2,2)
    img2 = a.imshow(np.real(P),cmap="coolwarm")
    a.set_title('Real part of $P_{suri}$')
    plt.colorbar(img2,orientation='horizontal')
    plt.savefig(outdir+'P_suri_visual.png')
    plt.clf()
    
    # Check norm of square of P
    P2 = np.dot(P,P)
    fig = plt.figure()
    a=fig.add_subplot(1,2,1)
    img = a.imshow(np.imag(P2),cmap="coolwarm")
    a.set_title('Imaginary part of $P^2_{suri}$')
    plt.colorbar(img, orientation ='horizontal')
    a=fig.add_subplot(1,2,2)
    img2 = a.imshow(np.real(P2),cmap="coolwarm")
    a.set_title('Real part of $P^2_{suri}$')
    plt.colorbar(img2,orientation='horizontal')
    plt.savefig(outdir+'P2_suri_visual.png')
    plt.clf()
    

# Check that P_ij is correct by transforming an eigenvector
fig = plt.figure()
a1=fig.add_subplot(2,2,1)
a1.set_title('Imaginary part')
a2=fig.add_subplot(2,2,2)
a2.set_title('Real part')
a3=fig.add_subplot(2,2,3)
a4=fig.add_subplot(2,2,4)
for i in [1,round(len(eigval)*0.5),round(len(eigval)*0.96)]:
    print 'eigv[', i, '] = ', eigval[i]
    #print 'eigvect[',i,'] = ', eigvect[i]
    #print 'eigvect shape = ', np.shape(eigvect[i])
    #print 'eigvect transpose shape = ', np.shape(eigvect[i].transpose())
    #print 'proj(eigv[',i,']) = ', eigvect[i]-np.dot( P, eigvect[i].transpose())
    a4.plot(np.arange(len(eigvect[i])),np.real(eigvect[i]),label=r'$\omega=$'+str(np.imag(eigval[i])))
    proj = np.dot( P, eigvect[i].transpose())
    a2.plot(np.arange(len(eigvect[i])),np.real(proj),label=r'$P e_{\omega}$, $\omega=$'+str(np.imag(eigval[i])))
    a3.plot(np.arange(len(eigvect[i])),np.imag(eigvect[i]),label=r'$\omega=$'+str(np.imag(eigval[i])))
    a1.plot(np.arange(len(eigvect[i])),np.imag(proj),label=r'$P e_{\omega}$, $\omega=$'+str(np.imag(eigval[i])))

# Shrink current axis by 20%
box = a1.get_position()
a1.set_position([box.x0-box.width*0.2, box.y0, box.width * 0.8, box.height])
box = a2.get_position()
a2.set_position([box.x0-box.width*0.4, box.y0, box.width * 0.8, box.height])
box = a3.get_position()
a3.set_position([box.x0-box.width*0.2, box.y0, box.width * 0.8, box.height])
box = a4.get_position()
a4.set_position([box.x0-box.width*0.4, box.y0, box.width * 0.8, box.height])

a2.legend(loc='center left', bbox_to_anchor=(1, 0.5),shadow=True, fancybox=True)
a4.legend(loc='center left', bbox_to_anchor=(1, 0.5),shadow=True, fancybox=True)
plt.savefig(outdir+'P_minus_Pe.png')
plt.clf()

# Now h_jkl = 12 pi i (P_jk P_kl P_lj - P_jl P_lk P_kj)
# Could buld up h_jkl here but instead just do the sum in nu below.
print 'build up h point by point'
h = np.zeros((len(P),len(P),len(P)),dtype=complex)
for j in range(len(P)):
    print ' summing j = ', j, ' of ', len(P)
    for k in range(len(P)):
        for l in range(len(P)):
            h[j,k,l] = P[j,k]*P[k,l]*P[l,j] - P[j,l]*P[l,k]*P[k,j]

h *= 12*np.pi*1j
print 'h = ', h
# Visualize h
fig = plt.figure()
a=fig.add_subplot(1,2,1)
img = a.imshow(np.imag(h[0]),cmap="coolwarm")
a.set_title('Imaginary part of $h_{0kl}$')
plt.colorbar(img, orientation ='horizontal')
a=fig.add_subplot(1,2,2)
img2 = a.imshow(np.real(h[0]),cmap="coolwarm")
a.set_title('Real part of $h_{0kl}$')
plt.colorbar(img2,orientation='horizontal')
plt.savefig(outdir+'h_visual.png')
plt.clf()

# Visualize h more
fig = plt.figure()
a=fig.add_subplot(1,2,1)
img = a.imshow(np.imag(h[50]),cmap="coolwarm")
a.set_title('Imaginary part of $h_{50,kl}$')
plt.colorbar(img, orientation ='horizontal')
a=fig.add_subplot(1,2,2)
img2 = a.imshow(np.real(h[50]),cmap="coolwarm")
a.set_title('Real part of $h_{50,kl}$')
plt.colorbar(img2,orientation='horizontal')
plt.savefig(outdir+'h_visual_more.png')
plt.clf()

# Divide plane into three, A->B->C counterclockwise
# First divide into two
reg1 = np.where(np.logical_and(xy[:,0]>-0.34,xy[:,1]>-0.45))[0]
regB = np.setdiff1d(np.arange(NP),reg1)
reg2IND = np.where(xy[regB,1]>-0.2)[0]
reg2 = regB[reg2IND]
reg3 = np.setdiff1d(regB,reg2)

# Check
plt.clf()
plt.scatter(xy[reg1,0],xy[reg1,1],c='k',edgecolor='none',alpha=0.5)
plt.scatter(xy[reg2,0],xy[reg2,1],c='g',edgecolor='none',alpha=0.5)
plt.scatter(xy[reg3,0],xy[reg3,1],c='r',edgecolor='none',alpha=0.5)

ax = plt.gca()
ax.set_title(r'Division of the lattice')
plt.savefig(outdir+'lattice_division.png')
#plt.show()

# Match dimensionality -- 2 dof per site
print 'reg1 = ', reg1
if args.XY_basis:
    reg1 = np.sort(np.vstack((2*reg1,2*reg1+1)).ravel())
    reg2 = np.sort(np.vstack((2*reg2,2*reg2+1)).ravel())
    reg3 = np.sort(np.vstack((2*reg3,2*reg3+1)).ravel())
else:
    print 'stacking regions with right-moving selves...'
    reg1 = np.sort(np.vstack((reg1,NP+reg1)).ravel())
    reg2 = np.sort(np.vstack((reg2,NP+reg2)).ravel())
    reg3 = np.sort(np.vstack((reg3,NP+reg3)).ravel())

print 'reg1 = ', reg1
print 'max(reg1) = ',max(reg1)
print 'max(reg2) = ',max(reg2)
print 'max(reg3) = ',max(reg3)
print '2*NP = ', 2*NP

# Then nu(P) = h(A,B,C) = sum_{j in A} sum_{k in B} sum_{l in C} h_{jkl}
print 'Summing up h values...'
#nu_steps = np.zeros(1000000, dtype=complex)
dmyi = 0
nu = 0.+0.*1j
for j in reg1:
    for k in reg2:
        for l in reg3:
            nu += h[j,k,l]
            #nu += 12*np.pi*1j*(P[j,k]*P[k,l]*P[l,j] - P[j,l]*P[l,k]*P[k,j])
            #nu_steps[dmyi] = nu
            dmyi +=1

    
print 'nu = ', nu
ax.set_title(r'$\nu=$ '+str(nu))
plt.savefig(outdir+'chern_no.png')

# Save steps
# nu_steps = nu_steps[0:dmyi]
# plt.clf()
# plt.plot(np.arange(dmyi), np.imag(nu_steps),label='imaginary part')
# plt.plot(np.arange(dmyi), np.real(nu_steps),label='real part')
# ax = plt.gca()
# ax.set_title(r'Evolution of $\nu$ during sum')
# ax.set_xlabel(r'Step')
# ax.set_ylabel(r'$\nu$')
# ax.legend(loc='best')
# plt.savefig(outdir+'nu_steps.png')
# plt.clf()


# # Test example
# u1 = np.array([3,2])
# u2 = np.array([-1,1])
# A = np.array([[2,3],[2,1]])
# U = np.array([[3,-1],[2,1]])
# U1 = la.inv(U)
# D = np.array([[4,0],[0,-1]])
# print '\n\n---------------------------\n\n'
# print 'np.dot(D,U1) = ', np.dot(U,np.dot(D,U1))
# print 'A - np.dot(U,np.dot(D,U1)) = ', A - np.dot(U,np.dot(D,U1))



