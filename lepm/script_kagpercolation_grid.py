import numpy as np
import lepm.lattice_elasticity as le
import lepm.gyro_lattice_functions as glatfns
import lepm.gyro_collection as gyro_collection
import matplotlib.pyplot as plt
import argparse
import lepm.lattice_class as lattice_class
import lepm.gyro_lattice_class as gyro_lattice_class
import lepm.plotting.plotting as leplt
import lepm.plotting.colormaps as cmaps
import lepm.plotting.bott_plotting_functions as bpfns
import lepm.plotting.bott_collection_plotting_functions as bcollpfns
import lepm.stringformat as sf
from lepm.bott.bott_gyro_collection import BottGyroCollection
import glob
import lepm.dataio as dio
try:
    import cPickle as pickle
except ImportError:
    import pickle
import os


"""
Script for computing bott index (or other topological invariant) as a function of kagomizing elements in
grid pieces of a lattice or amorphous network

$ python ./build/make_lattice.py -LT kagpergrid_hex -N 10 -periodic -alph 5.0 -perd 0.5

$ python run_series.py  -pro ./build/make_lattice -opts LT/kagpergrid_hex/-N/11/-periodic/-alph/5.0/-skip_polygons/-skip_gyroDOS -var perd 0.0:0.1:1.1
or
$ python run_series.py -pro ./build/make_lattice -opts LT/kagper_hex/-N/11/-periodic/-skip_polygons/-skip_gyroDOS -var perd 0.0:0.1:1.1

# Now make bott plot
python ./bott/bott_gyro_collection.py -vary_lpparam -lpparam percolation_density -LT kagper_hex -N 11

"""


lt = 'kagpergrid_hex'  # 'kagper_hex'
shape = 'square'
NH, NV = 11, 11
NP_load = 0
# alph is the grid size if lt == kagpergrid_hex
alph = 5.0

lp = {'LatticeTop': lt,
      'shape': shape,
      'NH': NH,
      'NV': NV,
      'NP_load': 0,
      'rootdir': '/Users/npmitchell/Dropbox/Soft_Matter/GPU/',
      'phi': 0.00 * np.pi,
      'delta': np.pi * 2./3.,
      'theta': 0.000,
      'eta': 0.000,
      'source': 'hexner',
      'loadlattice_number': 0,
      'check': check,
      'make_slit': False,
      'cutz_method': None,
      'cutLfrac': 0.0,
      'conf': 1,
      'periodicBC': True,
      'periodic_strip': False,
      'alph': alph,
      'origin': np.array([0., 0.]),
      'thres': 0.0,
      'spreading_time': 0.30,
      'spreading_dt': 0.001,
      'kicksz': -1.0,
      'aratio': 0.0,
      }

# Collate botts for one lattice with a gyro_lattice parameter that varies between instances of that lattice
meshfn = le.find_meshfn(lp)
lp['meshfn'] = meshfn
lat = lattice_class.Lattice(lp)
lat.load()
glat = gyro_lattice_class.GyroLattice(lat, lp)
glat.load()
gc = gyro_collection.GyroCollection()
gc.add_gyro_lattice(glat)
print 'Creating bott collection from single-lattice gyro_collection...'
kcoll = BottGyroCollection(gc, cp=cp)

# get the paramV from the available configurations on the hard disk
searchstr = lp['rootdir'] + 'networks/' + lt + '/' + lt + \
            '_square_periodicBC_delta0p667_phi0p000_perd*_{0:06d}'.format(NH) +\
            '_x_' + '{0:06d}'.format(NV) + '_xy.txt'
xystr = glob.glob(searchstr)
datadir
paramV = sf.string_sequence_to_numpy_array(args.paramV, dtype=args.paramVdtype)
kcoll.calc_botts_vary_glatparam(args.glatparam, paramV, max_ksize_frac=None, max_ksize=None, reverse=False,
                                verbose=False)
kcoll.plot_botts_vary_param(param=args.glatparam, param_type='glat')