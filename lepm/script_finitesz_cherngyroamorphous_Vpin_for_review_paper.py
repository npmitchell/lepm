import numpy as np
import lepm.lattice_elasticity as le
import lepm.dataio as dio
import lepm.data_handling as dh
import lepm.lattice_class as lattice_class
import lepm.gyro_lattice_class as gyro_lattice_class
import lepm.kitaev.kitaev_chern_class as kitaev_chern_class
import lepm.collapse_curves as coc
import lepm.plotting.plotting as leplt
import lepm.plotting.science_plot_style as sps
import lepm.plotting.colormaps as lecmaps
import lepm.plotting.colormaps
import lepm.kitaev.kitaev_functions as kfns
import lepm.stringformat as sf
import matplotlib.pyplot as plt
from scipy import optimize
import sys
import copy
import glob

"""
Not finished
Plot the Chern number results of hexagonal gyro networks while varying onsite disorder.
In this code, arrays are built in a way which would make it simple to also vary another parameter like ABDelta
in addition --> for ex, the nuchern array is 2D with dims len(NNlist) x 1 instead of being 1D.
"""

# Global plotting parameters
x0frac = 0.18
ortho = False
basis = 'XY'

#############################################################################################
# Plot the chern number convergence scaling with system size for disordered hexagonal gyro model networks
#############################################################################################
# TODO
# latticetop = 'randorg_gammakick0p20_cent'
# spreadingt = 0.
# NNlist = [64, 100, 225]  #, 400, 900]  # , 1225]  # , 1600]

# TODO
# latticetop = 'randorg_gammakick0p20_cent'
# spreadingt = 0.3
# NNlist = [64, 100, 225, 400, 900]  # , 1225]  # , 1600]

# TODO
# latticetop = 'randorg_gammakick0p50_cent'
# spreadingt = 0.
# NNlist = [64, 100, 225, 400, 900, 1225]  # , 1600]
#
# TODO
latticetop = 'randorg_gammakick0p50_cent'
spreadingt = 0.3
NNlist = [64, 100, 225, 400, 900, 1600, 2500]  # , 1225]
#
# TODO
# latticetop = 'randorg_gammakick0p80_cent'
# spreadingt = 0.
# NNlist = [64, 100, 225, 400, 900]  # , 1225]  # , 1600]
#
# TODO
# latticetop = 'randorg_gammakick0p80_cent'
# spreadingt = 0.3
# NNlist = [64, 100, 225, 400, 900]  # , 1600]  # , 1225]  # , 1600]


vpinlist_a = np.arange(0, 1.5, 0.1)
# vpinlist_b = np.arange(0.55, 1.00, 0.1)
# Vpinlist = np.sort(np.hstack((vpinlist_a, vpinlist_b, vpinlist_c, vpinlist_d, vpinlist_e, vpinlist_f)))
Vpinlist = vpinlist_a
print 'Vpinlist = ', Vpinlist
npinconfs = 10
nconfs = 10

fontsize = 8
ymax = 1.1
rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/'
cprootdir = '/Volumes/research4TB/Soft_Matter/GPU/'
rootroot = './data_for_figs/'
lepm.plotting.colormaps.register_colormaps()
lecmaps.register_colormaps()
# plt.get_cmap('BlueBlackRed')
rbb = plt.get_cmap('rbb0')
bbr = plt.get_cmap('bbr0')

lp = {'LatticeTop': latticetop,
      'shape': 'square',
      'rootdir': '/Users/npmitchell/Dropbox/Soft_Matter/GPU/',
      'delta_lattice': '0.667',
      'phi_lattice': '0.000',
      'theta': 0.,
      'eta': 0.,
      'x1': 0.,
      'x2': 0.,
      'x3': 0.,
      'z': 0.,
      'source': 'hexner',
      'loadlattice_number': 01,
      'check': False,
      'Ndefects': 0,
      'Bvec': 'W',
      'dislocation_xy': '0/0',
      'target_z': 0.,
      'make_slit': False,
      'cutz_method': 'none',
      'cutLfrac': 0.0,
      'conf': 01,
      'subconf': 01,
      'periodicBC': False,
      'loadlattice_z': '001',
      'alph': 1.0,
      'origin': np.array([0., 0.]),
      'Omk': -1.0,
      'Omg': -1.0,
      'V0_spring_gauss': 0.,
      'percolation_density': 0.5,
      'save_pinning_to_hdf5': True,
      'spreading_time': spreadingt,
      'kicksz': -1.5,
      'ortho': ortho,
      'basis': basis,
      }

cp = {'ksize_frac_arr': np.arange(0.0, 0.51, 0.01),  # sf.string_sequence_to_numpy_array('0.0:0.01:0.70', dtype=float),
      'omegac': 2.25,
      'shape': 'square',
      'polyT': False,
      'poly_offset': 'none',
      'basis': 'XY',
      'modsave': 10,
      'save_ims': False,
      'rootdir': cprootdir,
      'ortho': ortho,
      }

eps = 1e-7
colorv = lecmaps.husl_palette(len(Vpinlist))

# set limits of delta for color limits based on delta
minvpin = np.min(np.array(Vpinlist))
maxvpin = np.max(np.array(Vpinlist))

kk = 0
for vpin in Vpinlist:
    if vpin > eps:
        confv = np.arange(0, npinconfs, dtype=int)
    else:
        confv = np.array([0])
    netconfv = np.arange(1, nconfs + 1)

    nuchern = np.zeros((len(NNlist), len(confv) * len(netconfv)), dtype=float)
    ksznumax = np.zeros((len(NNlist), len(confv) * len(netconfv)), dtype=float)
    kszfracnumax = np.zeros((len(NNlist), len(confv) * len(netconfv)), dtype=float)
    nsites_arr = np.zeros((len(NNlist), len(confv) * len(netconfv)), dtype=int)
    nn_arr = np.zeros((len(NNlist), len(confv) * len(netconfv)), dtype=int)
    netconf_arr = np.zeros((len(NNlist), len(confv) * len(netconfv)), dtype=int)
    pinconf_arr = np.zeros((len(NNlist), len(confv) * len(netconfv)), dtype=int)
    colors = []
    # lp['latticetop'] = lt
    ii = 0
    for NN in NNlist:
        jj = 0
        dmyk = 0
        for netconf in netconfv:
            print 'considering network configuration = ', netconf
            for pinconf in confv:
                lpnew = copy.deepcopy(lp)
                cpnew = copy.deepcopy(cp)

                lpnew['NH'] = int(np.sqrt(NN))
                lpnew['NV'] = int(np.sqrt(NN))
                lpnew['NP_load'] = NN
                lpnew['conf'] = netconf
                lpnew['pinconf'] = pinconf
                lpnew['V0_pin_gauss'] = vpin

                # cpnew['omegac'] = bcfns.gap_midpoints_honeycomb(delta * np.pi)

                lat = lattice_class.Lattice(lp=lpnew)

                try:
                    lat.load()
                except IOError:
                    lat.build()
                    lat.save(skip_polygons=True)

                nsitesii = len(lat.xy)
                print 'kk, ii, jj, dmyk = ', kk, ii, jj, dmyk
                print 'vpin->', vpin, ' NN->', NN

                glat = gyro_lattice_class.GyroLattice(lat, lp=lpnew)
                chern = kitaev_chern_class.KitaevChern(glat, cp=cpnew)
                chern.get_kitaev_chern(skip_paramsregs=True)