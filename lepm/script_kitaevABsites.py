import numpy as np
import subprocess
import lepm.lattice_elasticity as le

# First reproduce the AB site diagram for the deformed honeycomb lattice
NN = 5  # 10
delta = np.arange(0.7, 1.31, 0.1)
delta = np.hstack((0.667, delta))
paramV = '0.0:0.1:1.0'
for delt in delta:
    subprocess.call(['python', 'kitaev_collection.py', '-vary_lpparam', '-LT', 'hexagonal', '-shape', 'hexagon',
                     '-lpparam', 'ABDelta', '-paramV', paramV, '-N', str(NN), '-delta', '{0:0.3f}'.format(delt)])


# Collect these cherns and plot them
lp = {'LatticeTop': 'hexagonal',
      'shape': 'hexagon',
      'NH': NN,
      'NV': NN,
      'NP_load': 0,
      'rootdir': '/Users/npmitchell/Dropbox/Soft_Matter/GPU/',
      'phi_lattice': '0.000',
      'delta': 2./3.*np.pi,
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
      'periodicBC': True,
      'loadlattice_z': '001',
      'alph': 1.0,
      'origin': np.array([0., 0.]),
      'Omk': -1.0,
      'Omg': -1.0,
      'V0_pin_gauss': 0.,
      'V0_spring_gauss': 0.,
      'dcdisorder': False,
      'percolation_density': 0.5,
      }

cp = {'ksize_frac_arr': le.string_sequence_to_numpy_array('0.0:0.01:1.10', dtype=float),
      'omegac': np.array([2.25]),
      'regalph': np.pi * (11. / 6.),
      'regbeta': np.pi * (7. / 6.),
      'reggamma': np.pi * 0.5,
      'shape': 'hexagon',
      'polyT': False,
      'poly_offset': 'none',
      'basis': 'XY',
      'modsave': 100,
      'save_ims': False,
      'rootdir': '/Volumes/research4TB/Soft_Matter/GPU/',
      }
