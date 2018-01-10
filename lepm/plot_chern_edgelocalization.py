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
import itertools

'''Plot the calculation of the chern number as summed region grows, plotting versus number of pts in sum
    This shows that the convergence is worse on edges where the confinement of edge modes is less confined (zigzag)
'''

marker = itertools.cycle((',', '+', '.', 'o', '*'))

parser = argparse.ArgumentParser(description='Specify parameters for computing Chern number in realspace Haldane model.')
parser.add_argument('-seriesdir','--seriesdir',help='Name for the directory in which to store plot and find data',
                    type=str, default='honeycomb_delta0p667')
parser.add_argument('-max_offset', '--max_offset', help='Maximum value for the offset of the summed region',
                    type=float, default=12.0)
args = parser.parse_args()


FSFS = 20
markers = ['o','s','*','^']
colors = ['#52B6F1','#D27348']
# deltaV = np.arange(0.750,1.30,0.1)
# deltaV = np.hstack((np.array([0.667]),np.array([0.995]), deltaV))
rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/experiments/kitaev_chern_gyro_finsize/edge_localization_effects/honeycomb_N20/'
seriesdir = rootdir + args.seriesdir + '/'
lepm.plotting.colormaps.register_colormaps()
cool = plt.get_cmap('cool')
# BBR = plt.get_cmap('BlueBlackRed')
# RBB = plt.get_cmap('RedBlackBlue')

# Sort the vectors
print 'seriesdir = ', seriesdir
subdirs = le.find_subdirs('20*', seriesdir)

ind = 0
numax = np.zeros(len(subdirs), dtype=float)
distance = np.zeros(len(subdirs), dtype=float)
polyTV = np.zeros(len(subdirs), dtype=int)
colorList = []
markerList = []
labelList = []
dimList = []
print 'markerList = ', markerList
for subdir in subdirs:
    print 'found subdir = ', subdir + 'chern_finsize.txt'
    N, Nparticles, Nreg1, ksize_frac, ksys_size, nu = np.loadtxt(subdir+'chern_finsize.txt',delimiter=',',unpack=True)
    params = le.load_params(subdir)
    poff = params['poly_offset']
    if 'polyT' in params:
        polyT = params['polyT']
    else:
        polyT = params['poly_rotate']
    print 'poff = ', poff
    if poff[0] > 1e-6:
        print 'true'
        dim = 0
        colorii = cool(poff[0] / args.max_offset)
        label = r'$\delta x$ ='+'{0:0.1f}'.format(poff[0])
        plt.plot(Nreg1 * 0.5, nu, 'o-', color=colorii, lw=2,  label=label)
        marker = 'o'
    elif polyT:
        dim = 1
        colorii = cool(poff[1] / args.max_offset)
        label = r'$\delta y$ =' + '{0:0.1f}'.format(poff[1]) + ' Transp.'
        plt.plot(Nreg1 * 0.5, nu, 's-', color=colorii, lw=2, label=label)
        marker = 's'
    elif poff[1] > 1e-6:
        dim = 1
        colorii = cool(poff[1] / args.max_offset)
        label = r'$\delta y$ =' + '{0:0.1f}'.format(poff[1])
        plt.plot(Nreg1 * 0.5, nu, '*-', color=colorii, lw=2, label=label)
        marker = '*'
    else:
        # Sample is centered and oriented normally
        dim = 0
        colorii = cool(poff[0] / args.max_offset)
        label = r'$\delta x$ =' + '{0:0.1f}'.format(poff[0])
        plt.plot(Nreg1 * 0.5, nu, 'o-', color=colorii, lw=2, label=r'$\delta x$ =' + '{0:0.1f}'.format(poff[0]))
        marker = 'o'

    # save max(|Chern|)
    imax = np.argmax(np.abs(nu))
    numax[ind] = nu[imax]

    # get distance from sample edge --> only works for rectangular samples (nondisordered)!
    ssd = le.find_subdirs('N*', subdir)
    xy = np.loadtxt(ssd[0] + 'xy.txt', delimiter=',')
    distance[ind] = np.max(xy[:, dim]) - poff[dim]
    colorList.append(colorii)
    markerList.append(marker)
    labelList.append(label)
    dimList.append(dim)
    polyTV[ind] = polyT
    ind += 1

plt.legend()
# plt.colorbar(cmap='RedBlackBlue')
plt.xlabel('number of points in Region A', fontsize=FSFS)
plt.ylabel('Chern number', fontsize=FSFS)
plt.title('Chern Number near edge', fontsize=FSFS+4)
plt.savefig(seriesdir + 'edgelocalization.png')


# Plot maxima vs distance
MM = np.dstack((distance, numax, np.array(dimList), polyTV))[0]
np.savetxt(seriesdir + 'nu_distance.txt', MM, delimiter=',',
           header='dist numax dim polyT : (distance from edge of rectangular sample, max chern number nu,' +\
                  'dimension of offset, sum polygon transposed or not')
plt.clf()
labeldone = {'o': False, 's': False, '*': False}
for ii in range(len(distance)):
    # Grab whether or not to label the point (only label the first time of its kind
    if labeldone[markerList[ii]]:
        label = ''
    else:
        splitlab = labelList[ii].split('=')
        lab_part2 = splitlab[1].split('.')[1]
        #try:
        print 'lab_part2 = ', lab_part2
        print 'lab_part2[1] = ', lab_part2[1:]
        label = splitlab[0] + lab_part2[1:]
        #except:
        #    label = splitlab[0]
        labeldone[markerList[ii]] = True

    # Plot the data as blue or purple (right edge or top edge)
    if dimList[ii] == 0:
        plt.plot(distance[ii], numax[ii], marker=markers[dimList[ii]+polyTV[ii]], color=cool(0.0), label=label)
    else:
        plt.plot(distance[ii], numax[ii], marker=markers[dimList[ii]+polyTV[ii]], color=cool(1.0), label=label)
plt.legend()
plt.xlabel(r'distance from edge', fontsize=FSFS)
plt.ylabel('Chern number', fontsize=FSFS)
plt.title('Chern Number near sample edge', fontsize=FSFS+4)
plt.savefig(seriesdir + 'edgelocalization_distance.png')
plt.close('all')

################################################
# Collate results for all seriesdirs
################################################
if '/' in seriesdir:
    # Have the path for subdir search be the whole seriesdir except for the end
    searchdir = ''
    for seg in seriesdir.split('/')[0:-2]:
        searchdir += seg + '/'
        print 'searchdir = ', searchdir
else:
    searchdir = rootdir
print 'searchdir =', searchdir
subdirs = le.find_subdirs(args.seriesdir[0:2]+'*', searchdir)

print 'found subdirs = ', subdirs
dmyi = 0
for subdir in subdirs:
    print 'collating with subdir = ', subdir
    # grad delta from name
    delta = subdir.split('/')[-2].split('delta')[-1].replace('p','.')

    # determine alpha from delta
    alpha = 1.0 - abs( (float(delta) - 0.667))/.28

    dist, nu, dim, polyT = np.loadtxt(subdir + 'nu_distance.txt', delimiter=',', unpack=True)

    # sort all arrays based on distance
    dist, [nu, dim, polyT] = le.sort_arrays_by_first_array(dist, [nu, dim, polyT])

    # discriminate regions which are on armchair (right) versus zigzag (top) and those which are transposed
    tIND_T = dim + polyT == 2
    tIND = dim + polyT == 1
    rIND = dim + polyT == 0
    plt.plot(dist[tIND_T], nu[tIND_T], '-', color=colors[1])

    plt.plot(dist[rIND], nu[rIND], '-', lw=2, color=colors[0], alpha=alpha)
    plt.plot(dist[rIND], nu[rIND], marker=markers[dmyi], color=colors[0], alpha=alpha,  label=r'$\delta =$'+delta+r'$\pi$ armchair')
    plt.plot(dist[tIND_T], nu[tIND_T], '-', lw=2, color=colors[1], alpha=alpha)
    plt.plot(dist[tIND_T], nu[tIND_T], marker=markers[dmyi], color=colors[1], alpha=alpha, label=r'$\delta =$'+delta+r'$\pi$ zig-zag')
    dmyi += 1

plt.legend(fontsize=FSFS-6)
plt.xlabel(r'distance from edge [lattice spacings]', fontsize=FSFS)
plt.ylabel(r'Chern number', fontsize=FSFS)
plt.title('Chern number near sample edge', fontsize=FSFS+4)
plt.savefig(searchdir+'edge_chern.pdf')
plt.show()


