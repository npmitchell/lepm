import numpy as np
import lepm.lattice_elasticity as le
import matplotlib.pyplot as plt
import make_lattice as makeL
import argparse
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
import sys
import glob
import lepm.plotting.colormaps

FSFS = 14
deltaV = np.arange(0.750,1.30,0.1)
deltaV = np.hstack((np.array([0.667]),np.array([0.995]), deltaV))
print 'deltaV = ', deltaV
rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/experiments/kitaev_chern_gyro_finsize/hexagonal_bowtie_transition/N15_eta0p60/'
lepm.plotting.colormaps.register_colormaps()
BBR = plt.get_cmap('BlueBlackRed')
RBB = plt.get_cmap('RedBlackBlue')
print 'RBB = ', RBB
print 'RBB(0.5) = ', RBB(0.5)


# Sort the vectors
sorti = np.argsort(deltaV)
deltaV = deltaV[sorti]
deltaList = deltaV.tolist()

ind = 0
numax = np.zeros(len(deltaV))
for delta in deltaList:
    deltastr = '{0:0.3f}'.format(delta)
    ddir = glob.glob(rootdir + '20*_delta' + deltastr.replace('.','p'))[0]
    print 'found ddir = ', ddir
    N, Nparticles, ksize_frac, ksys_size, nu = np.loadtxt(ddir+'/chern_finsize.txt',delimiter=',',unpack=True)
    colorii = RBB((delta - np.min(deltaV))/(np.max(deltaV)-np.min(deltaV)))
    plt.plot(ksys_size/Nparticles*0.5, nu, 'o-', color=colorii, lw=3,  label=r'$\delta$ ='+deltastr)
    
    # save max(|Chern|)
    imax = np.argmax(np.abs(nu))
    numax[ind] = nu[imax]
    ind += 1

plt.legend()
# plt.colorbar(cmap='RedBlackBlue')
plt.xlabel('Fraction of system in sum', fontsize=FSFS)
plt.ylabel('Chern number', fontsize=FSFS)
plt.title('Chern Number for Honeycomb Transition', fontsize=FSFS+4)
plt.savefig(rootdir + 'transition.png')


# Plot maxima vs delta
# First sort maxima according to increasing delta
# sorti = np.argsort(deltaV)
# deltaV = deltaV[sorti]
# numax = numax[sorti]
MM = np.dstack((deltaV, numax))[0]
np.savetxt(rootdir + 'nu_delta.txt', MM, delimiter=',', header='deformation angle delta vs max chern number nu')
plt.clf()
plt.plot(deltaV, numax, 'o-', color=colorii, lw=3)
plt.xlabel(r'$\delta$', fontsize=FSFS)
plt.ylabel('Chern number', fontsize=FSFS)
plt.title('Chern Number for Honeycomb Transition', fontsize=FSFS+4)
plt.savefig(rootdir + 'transition_delta.png')


# Plot chern number versus delta
rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/experiments/kitaev_chern_gyro_finsize/hexagonal_bowtie_transition/'
NstrList = [ '10', '15' ]
etaV = np.array([0.0,0.6])
etaList = etaV.tolist()

plt.clf()
for eta in etaList:
    for Nstr in NstrList:
        etastr = '{0:0.2f}'.format(eta)
        ddirs = glob.glob(rootdir + 'N' + Nstr + '*' + etastr.replace('.','p'))
        try:
            ddir = ddirs[0]

            delta, nu = np.loadtxt(ddir + '/nu_delta.txt', delimiter=',', unpack=True)
            colorii = RBB((eta - np.min(etaV)) / (np.max(etaV) - np.min(etaV)))
            Nstrii_tmp = ddir.split('N')[-1]
            Nstrii = Nstrii_tmp.split('_')[0]

            if int(Nstrii) > 10:
                marker = '^-'
            else:
                marker = 'o-'
            plt.plot(delta, nu, marker, color=colorii, lw=1, label=r'$N = $'+Nstrii + ' $\eta$ =' + etastr)
        except:
            print 'Could not find configuration...'

plt.legend()
plt.xlabel(r'$\delta$', fontsize=FSFS)
plt.ylabel('Chern number', fontsize=FSFS)
plt.title('Chern Number for Honeycomb Transition', fontsize=FSFS+4)
plt.savefig(rootdir + 'transition_delta_eta.png')