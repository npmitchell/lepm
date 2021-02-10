from lepm.lattice_class import Lattice
import lepm.lattice_elasticity as le
import numpy as np

rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/'

lp = {'LatticeTop': 'custom',
      'shape': 'circle',
      'NH': 10,
      'NV': 10,
      'NP_load': 0,
      'rootdir': '/Users/npmitchell/Dropbox/Soft_Matter/GPU/',
      'phi': 0.,
      'delta': '0.667',
      'theta': 0.,
      'eta': 0.,
      'make_slit': False,
      'cutz_method': 'none',
      'cutLfrac': 0.0,
      'conf': 01,
      'periodicBC': False,
      'alph': 0.,
      'origin': np.array([0., 0.]),
      'thres': 0,
      'spreading_time': 0.,
      'kicksz': -1.5,
      'lattice_exten': '_custom',
      }

presfn = '/Users/npmitchell/Dropbox/Soft_Matter/Presentations/2017_MarchMeeting/'
xyfn = presfn + 'coords.txt'
xy = np.loadtxt(xyfn)
xy, NL, KL, BL, BM = le.delaunay_lattice_from_pts(xy, check=True, max_bond_length=55)
polygons = le.extract_polygons_lattice(xy, BL, NL=NL, KL=KL, check=True)
lat = Lattice(lp, xy=xy, NL=NL, KL=KL, BL=BL, polygons=polygons)
print 'Saving nice WB plot... (white-black)'
lat.plot_WB_lat(meshfn=presfn, lw=2)
