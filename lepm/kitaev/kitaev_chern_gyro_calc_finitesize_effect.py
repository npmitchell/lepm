import numpy as np
import numpy.linalg as la
import lepm.lattice_elasticity as le
import lepm.lattice_class as lattice_class
import matplotlib.pyplot as plt
import make_lattice as makeL
import copy
import argparse
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
import sys
import lepm
import lepm.plotting.colormaps
import pickle
import cPickle
import glob
import polygon_functions as polyfns

'''Calculate Chern number via realspace method Kitaev method for
a range of system sizes (or just one).

Usage ex:
python kitaev_chern_gyro_calc_finitesize_effect.py -LT hexagonal -N 20 -shape hexagon -ksize 0.1:0.1:1.2 -skip_DOS_ims
python kitaev_chern_gyro_calc_finitesize_effect.py -LT hexagonal -N 20 -shape hexagon -ksize 0.1:0.1:1.2 -skip_DOS_ims
python kitaev_chern_gyro_calc_finitesize_effect.py 20160415-2059 -store_DOS -LT dislocated -N 6 -shape hexagon -ksize 0.1:0.1:1.1
python kitaev_chern_gyro_calc_finitesize_effect.py 20160415-2059 -LT hexagonal -N 6 -shape hexagon -ksize 0.1:0.1:1.2 -skip_DOS_ims
python kitaev_chern_gyro_calc_finitesize_effect.py 20160415-2059 -LT honeycomb -eta 0.6 -N 10 -shape hexagon -ksize 0.1:0.2:.4

python kitaev_chern_gyro_calc_finitesize_effect.py -LT hucentroid -N 50 -shape square -omegac 2.245 -ksize 0.0:0.05:1.2
python kitaev_chern_gyro_calc_finitesize_effect.py -LT kagome_hucent -N 50 -shape square -omegac 2.200 -sqrt_ksizestep -ksize 0.0:0.01:1.2
python kitaev_chern_gyro_calc_finitesize_effect.py -LT kagome_penroserhombTricent -N 10 -shape circle -omegac 2.250 -sqrt_ksizestep -ksize 0.0:0.01:1.2

# Percolation
python kitaev_chern_gyro_calc_finitesize_effect.py -LT kagper_hucent -N 30 -shape square -perd 0.1 -omegac 2.250 -sqrt_ksizestep -ksize 0.0:0.01:1.2

# Periodic samples
python kitaev_chern_gyro_calc_finitesize_effect.py -periodic -N 6 -ksize 0.0:0.1:1.2
'''


lepm.plotting.colormaps.register_colormaps()

parser = argparse.ArgumentParser(description='Specify parameters for computing Chern number in realspace Haldane model.')
parser.add_argument('datehourmin', type=str, nargs='?',
                    help='Name of simulation to build, in format YYYYMMDD-HHMM',
                    default='check_string_for_empty')
parser.add_argument('-seriesdir', '--seriesdir',help='Name for the directory in which to store the calculation datadir', type=str, default='kitaev_chern_gyro_finsize')
parser.add_argument('-basis', '--basis',help='Basis in which to compute the eigenvectors, projector, etc', type=str, default='XY')
parser.add_argument('-periodic', '--periodicBC', help='Enforce periodic boundary conditions', action='store_true')
parser.add_argument('-skip_ims', '--skip_ims', help='Do not save DOS or region images', action='store_true')
parser.add_argument('-check', '--check', help='Show progress images during calculation', action='store_true')
parser.add_argument('-checkout', '--checkout', help='Save progress images during calculation', action='store_true')
parser.add_argument('-Omg','--Omg', help='Frequency of gravitational precession (specify if homogenous across sample)', type=float, default=-1.0)
parser.add_argument('-Omk','--Omk', help='Frequency of spring interaction (specify if homogenous across sample)', type=float, default=-1.0)
parser.add_argument('-ksize','--ksize_frac', help='Fraction of system size to include in kitaev calculation. Specify a series of values if desired',
                    type=str, default='0.0:0.01:1.2')
parser.add_argument('-sqrt_ksizestep', '-sqrt_ksizestep', help='Use square root of sample size fractions to sum', action='store_true')
parser.add_argument('-skip_DOS_ims','--skip_DOS_ims', help='Skip plotting and saving Density Of States results', action='store_true')
parser.add_argument('-save_DOS','--save_DOS', help='Save Density Of States results as pickle files', action='store_true')

# Kitaev params
parser.add_argument('-poly_offset','--poly_offset', help='Offset for the center of the polygon', type=str, default='none')
parser.add_argument('-polyT','--polyT', help='Transpose polygon', action='store_true')
parser.add_argument('-omegac','--omegac', help='Frequency cutoff for chern number calculation', type=float, default=-500)

# Geometry
parser.add_argument('-slit','--slit', help='Declare whether or not there is a slit', action='store_true')
parser.add_argument('-N','--N', help='Mesh width AND height, in number of lattice spacings. Specify a series of values by using / between vals (ex 3/4/5/6)',
                    type=str, default='5')
parser.add_argument('-NH','--NH', help='Mesh width, in number of lattice spacings', type=int, default=10)
parser.add_argument('-NV','--NV', help='Mesh height, in number of lattice spacings', type=int, default=10)
parser.add_argument('-NP','--NP_load',
                    help='Number of particles in mesh, overwrites N, NH, and NV.',
                    type=int, default=0)
parser.add_argument('-LT','--LatticeTop', \
                    help='Lattice topology: linear, hexagonal, triangular, deformed_kagome, twisted_kagome, square, hyperuniform, hyperuniformdual, isostatic', \
                    type=str, default='hexagonal')
parser.add_argument('-delta','--delta_lattice', help='for hexagonal lattice, measure of deformation in radians/pi', type=str, default='0.667')
parser.add_argument('-phi','--phi_lattice', help='for hexagonal lattice, measure of deformation in radians/pi', type=str, default='0.000')
parser.add_argument('-eta','--eta', help='Lattice randomization/jitter (usually units of unity/lattice spacing)', type=float, default=0.000)
parser.add_argument('-theta','--theta', help='Lattice rotation (units of pi radians)', type=float, default=0.000)
parser.add_argument('-x1','--x1', help='Deformed kagome geometric parameter', type=float, default=0.000)
parser.add_argument('-x2','--x2', help='Deformed kagome geometric parameter', type=float, default=0.000)
parser.add_argument('-x3','--x3', help='Deformed kagome geometric parameter', type=float, default=0.000)
parser.add_argument('-zkagome','--zkagome', help='Deformed kagome geometric parameter', type=float, default=0.000)
parser.add_argument('-cutL','--cutL', help='String specifying length of slit as fraction of L', type=str, default='0.5')
parser.add_argument('-shape','--shape', help='Shape of the overall mesh geometry', type=str, default='hexagon')
parser.add_argument('-huID','--hyperuniform_number', help='If LT=hyperuniform, selects which hyperuniform lattice to use', type=str, default='01')
parser.add_argument('-z','--target_z', help='Average coordination of the lattice to load. If -1, loads default coord', type=str, default='-1.')
parser.add_argument('-cutz_method','--cutz_method', help='random or highest, method by which z target value is obtained', type=str, default='highest')
parser.add_argument('-Ndefects', '--Ndefects', help='Number of defects to introduce', type=int, default = 1)
parser.add_argument('-Bvec', '--Bvec', help='Direction of burgers vectors of dislocations (random, W, SW, etc)', type=str, default = 'W')
parser.add_argument('-dislocxy', '--dislocation_xy',help='Position of single dislocation, if not centered at (0,0), as strings sep by / (ex: 1/4.4)', type=str, default = 'none')
parser.add_argument('-deform','--deform', help='Whether to deform the lattice over time in sim', action='store_true')
parser.add_argument('-deform_rate','--deform_rate', help='Rate in radians/unit time to deform lattice', type=float, default=0.001)
parser.add_argument('-adiabatic','--deform_adiabatic', help='Whether to drag xy along with xy0 in deformation', action='store_true')
parser.add_argument('-perd','--percolation_density', help='Density of vertices decorated in percolation network', type=float, default=0.50)
parser.add_argument('-alph','--tk_alph', help='Twisted kagome twist angle, in radians', type=float, default=0.000)
parser.add_argument('-alph_final','--alph_final', help='Final deformation parameter in realtime deform (for twisted_kagome, this is pi/3)',\
                    type=float, default=1.0471975511965976)
args = parser.parse_args()


FSFS = 20
maindir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/experiments/'
# Get timestamp for this calculation
print '\n args.datehourmin = ', args.datehourmin
timestr, newdatadir, LOADPARAMS, datadir = le.determine_simulation_timestamp(args.datehourmin, maindir)
datadir = maindir+args.seriesdir+'/'+timestr
if args.delta_lattice != '0.667':
    datadir += '_delta' + args.delta_lattice.replace('.','p')
if args.LatticeTop != 'hexagonal':
    datadir += '_'+args.LatticeTop

# Determine omegac and add it to the name
if args.omegac != -500:
    omegac = args.omegac
else:
    if args.LatticeTop == 'hexagonal':
        if np.abs(args.Omg) == 1 and np.abs(args.Omk) == 1:
            # load gap edge from txt file
            deltas, maxes = np.loadtxt(maindir + 'gap_edges/' + args.LatticeTop + '/maxes.txt', unpack=True)
            ind = np.argmin( np.abs(deltas - float(args.delta_lattice)*180. ))
            print 'ind = ', ind
            print 'maxes[ind] = ', maxes[ind]
            omegac = maxes[ind]
            print 'determined omegac = ', omegac
        else:
            print 'Not sure what omegac to default to...'
            sys.exit()
    else:
        if 'triangular' in args.LatticeTop or args.LatticeTop in ['isostatic', 'jammed']:
            print 'Lattice is triangular: assuming gap is at zero frequency!'
            omegac = 0.0
        else:
            print 'Assuming gap is at 2.25 inverse time.'
            omegac = 2.25

datadir += '_omegac' + '{0:0.2f}'.format(omegac).replace('.','p')
if args.poly_offset != 'none':
    if '/' in args.poly_offset:
        datadir += '_offx' + args.poly_offset.split('/')[0].replace('.','p') + '_y' + args.poly_offset.split('/')[1].replace('.','p')
    else:
        datadir += '_offx' + args.poly_offset.split('_')[0].replace('.', 'p') + '_y' + args.poly_offset.split('_')[1].replace('.', 'p')
if args.polyT:
    datadir += '_polyT'

datadir += '/'

# Series of sizes to run is N array (Narr) or series of other values
Narr = le.string_sequence_to_numpy_array(args.N, dtype = int)
ksize_arr = le.string_sequence_to_numpy_array(args.ksize_frac, dtype = float)
if args.sqrt_ksizestep:
    ksize_arr = np.sqrt(ksize_arr)

print 'Narr = ', Narr
print 'ksize_arr = ', ksize_arr
print 'type(Narr) = ', type(Narr)
sysV = np.zeros(len(Narr)*len(ksize_arr), dtype=int)
Nreg1V = np.zeros(len(Narr)*len(ksize_arr), dtype=int)
N_V = np.zeros(len(Narr)*len(ksize_arr), dtype=int)
nuV = np.zeros(len(Narr)*len(ksize_arr), dtype=complex)
ksize_V = np.zeros(len(Narr)*len(ksize_arr), dtype=float)
ksys_sizeV = np.zeros_like(ksize_V)
 
# parameters
LatticeTop = args.LatticeTop
periodic = args.periodicBC
shape = args.shape
save_ims = not args.skip_ims
save_DOS_ims = not args.skip_DOS_ims
save_DOS_pkl = args.save_DOS 
check = args.check
checkout = args.checkout

regalph = np.pi*(11./6.)
regbeta = np.pi*(7./6.)
reggamma = np.pi*0.5
if args.poly_offset != 'none':
    if '/' in args.poly_offset:
        splitpo = args.poly_offset.split('/')
    else:
        splitpo = args.poly_offset.split('_')
    print 'split_po = ', splitpo
    poly_offset = np.array([float(splitpo[0]), float(splitpo[1])])
else:
    poly_offset = np.array([0., 0.])

# Use each value of N
for jj in range(len(Narr)):
    N = Narr[jj]
    print '\nN = ', N
    dataNdir = datadir+'N_{0:03d}'.format(N)+'/'
    dio.ensure_dir(dataNdir)
    NH = N
    NV = N
    Ns = min(NH, NV)
    
    # Get polygon over which to sum
    if LatticeTop in ['triangular', 'triangularz']:
        dist = 1.
    #else:
    #    # distance for NH=1 is dist
    #    if args.delta == '0.667':
    #        dist = 2.*le.polygon_apothem(1.0,6) # distance across one hexagon of bonds with bondlength=1
    #    else:
    #        a1, a2 = deformed_hexcell_to_hexagonal_sidelengths(args.delta,args.NH,args.NV)

    # Make Lattice
    sourcedir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU'
    eta = args.eta
    if args.slit:
        cutLstr   = '_cutL'+args.cutL+'L_x0_*'  # '_cutL0.2L_x0_*' ''\
    else:
        cutLstr = ''
    lp = {'LatticeTop': args.LatticeTop,
          'periodicBC': args.periodicBC,
          'shape': shape,
          'rootdir': sourcedir,
          'NH': NH,
          'NV': NV,
          'NP_load': args.NP_load,
          'x1': args.x1,
          'x2': args.x2,
          'x3': args.x3,
          'zkagome': args.zkagome,
          'alph': args.tk_alph,
          'periodic': args.periodicBC,
          'cutLstr': cutLstr,
          'delta_lattice': args.delta_lattice,
          'phi_lattice': args.phi_lattice,
          'theta_lattice': args.theta,
          'eta': args.eta,
          'huID': args.hyperuniform_number,
          'z': args.target_z,
          'Ndefects': args.Ndefects,
          'Bvec': args.Bvec,
          'dislocxy': (args.dislocation_xy.split('/')),
          'cutz_method': args.cutz_method,
          'origin': np.array([0.,0.]),
          'source': 'hexner',
          'percolation_density': args.percolation_density,
    }

    lat = lattice_class.Lattice(lp=lp, xy=np.array([]), NL=np.array([]), KL=np.array([]), BL=np.array([]),
                                polygons=None)
    lat.load()

    # xyload, NLload, KLload, meshfn = le.find_lattice(lp)
    # xy = xyload.astype(float)
    # NL = NLload.astype(int)
    # KL = KLload.astype(int)
    # BL = le.NL2BL(NL, KL)
    xy = lat.xy
    NL = lat.NL
    KL = lat.KL
    BL = lat.BL
    meshfn = lat.lp['meshfn']

    PVxydict = lat.PVxydict
    # le.load_evaled_dict(le.prepdir(meshfn),'PVxydict')
    # except:
    #    PVxydict = []
    try:
        zcoord = le.compute_bulk_z(xy,NL,KL)
    except:
        print 'Could not compute coordination (perhaps too few pts not on boundary). \nSetting z=0.'
        zcoord = 0
    OmK = args.Omk * KL.astype(float)
    Omg = args.Omg * np.ones_like(xy[:,0])
    
    params = {}
    NP = len(xy)
    
    # Make or load Dynamical Matrix
    if args.basis == 'XY':
        # Check if eigvect, eigval, DM exist
        print '/n/n/n find-->', glob.glob(meshfn + '/eigval.pkl')
        eigvalfn = glob.glob(meshfn + '/eigval.pkl')
        eigvectfn = glob.glob(meshfn + '/eigvect.pkl')
        OmgOmk_ok = (args.Omg == -1.0 and args.Omk == -1.0)
        print 'eigvalfn = ', eigvalfn
        if eigvalfn and eigvectfn and OmgOmk_ok:
            print '\n\n found eigval, eigvect. Loading...'
            with open(eigvalfn[0], "rb") as input_file:
                eigval = cPickle.load(input_file)
            with open(eigvectfn[0], "rb") as input_file:
                eigvect = cPickle.load(input_file)
        else:
            print '\n\n Did not find eigval, eigvect. Computing...'
            eigvect, eigval, DM = le.plot_save_normal_modes_Nashgyro(dataNdir,xy, NL, KL, OmK, Omg, params={},
                                                                     save_pkl= save_DOS_pkl, save_ims=save_DOS_ims, gapims_only=False)
            if OmgOmk_ok:
                print '\n\n Saving computed eigval, eigvect...'
                output = open(meshfn + '/eigval.pkl', 'wb')
                pickle.dump(eigval, output)
                output.close()
                output = open(meshfn + '/eigvect.pkl', 'wb')
                pickle.dump(eigvect, output)
                output.close()

    elif args.basis == 'psi':
        eigvect, eigval, DM = le.save_normal_modes_Nashgyro_psirep(dataNdir,xy, NL, KL, OmK, Omg, params={},
                                                                   save_pkl = save_DOS_pkl, save_ims=save_DOS_ims, gapims_only=False)

    # print 'eigval = ', eigval
    # print 'eigvect = ', eigvect
    U = eigvect.transpose()
    U1 = la.inv(U)
    D = np.zeros((len(U), len(U)), dtype=complex)
    
    for ii in range(len(eigval)):
        ev = eigval[ii]
        D[ii, ii] = ev

    # M = copy.deepcopy(D)
    # M[M.imag < omegac] = 0
    # M[M.imag > omegac] = 1

    # Try projecting onto a single state
    M = np.zeros_like(D, dtype=float)
    M[100, 100] = 1

    P = np.dot(U, np.dot(M, U1))
    h = np.zeros((len(P), len(P), len(P)), dtype=complex)
    
    ######################################################
    # CHECK
    if check or checkout:
        le.plot_complex_matrix(DM,name='Dynamical Matrix',outpath=dataNdir+'DM',show=check,close=True)
        le.plot_complex_matrix(D,name='Eigenvalue Matrix',outpath=dataNdir+'D',show=check,close=True)
        le.plot_complex_matrix(M,name='Cut-Off Eigval Matrix',outpath=dataNdir+'M',show=check,close=True)
        le.plot_complex_matrix(P,name='Projector',outpath=dataNdir+'P',show=check,close=True)
        # P is hermitian
        le.plot_complex_matrix(P-P.transpose().conjugate(),name=r'$P -P^{dagger}$',outpath=dataNdir+'PminusPdagger',show=check,close=True)
        # P squares to unity
        le.plot_complex_matrix(np.dot(P,P),name=r'$P^2$',outpath=dataNdir+'P2',show=check,close=True)
        # Check that P_ij is correct by transforming an eigenvector to itself or to zero
        inds2proj = [1,round(len(eigval)*0.5),round(len(eigval)*0.96)]
        le.plot_projection_complex_vects(P,eigvect,eigval,inds2proj,fig='auto', outpath=dataNdir+'Peigvects', show=check,close=True)
    ######################################################
    
    # Save params for this size
    params = {'date'      : timestr   ,
              'meshfn'    : meshfn    ,
              'shape'     : shape     ,
              'NP'        : NH        ,
              'NH'        : NH        ,
              'NV'        : NV        ,
              'datadir'   : datadir   ,
              'LatticeTop': LatticeTop,
              'NL'        : NL        ,
              'KL'        : KL        ,
            'xy'        : xy        , #undeformed lattice sites
            'OmK'       : OmK       ,
            'Omg'       : Omg       ,
            'z'         : zcoord    ,
            'periodicBC'  : periodic  ,
            'eta'       : eta       ,
            }
    le.write_initparams(params,dataNdir)

    # Initialize empty region index arrays for speedup by comparing with prev iteration
    reg1 = np.array([])
    reg2 = np.array([])
    reg3 = np.array([])
    reg1_xy = np.array([])
    reg2_xy = np.array([])
    reg3_xy = np.array([])
    nu = 0.0 + 0.0 * 1j
    epskick = 0.001*np.random.rand(len(xy),2)

    # for each ksize_frac, perform sum
    for kk in range(len(ksize_arr)):
        ksize_frac = ksize_arr[kk]
        
        # Divide plane into three, A->B->C counterclockwise
        teps = 1e-7 #small offset of region boundaries
        if args.shape == 'hexagon':
            # ksize = Ns*ksize_frac*dist
            delta = np.pi * float(args.delta_lattice)
            a1, a2 = polyfns.deformed_hexcell_to_hexagonal_sidelengths(delta,ksize_frac*NH.astype(float), ksize_frac*NV.astype(float))
            print 'a1, a2 = ', a1, a2
            hexagon = polyfns.deformed_hex_polygon(a1, a2)
            print 'hexagon = ', hexagon
            # plt.clf()
            # plt.plot(hexagon[:,0],hexagon[:,1],'r-')
            # plt.show()
            polygon1, polygon2, polygon3 = polyfns.divide_hexagon_by_sidelengths(hexagon, eps=1e-8)
            print 'polygon1 = ', polygon1
            print 'polygon2 = ', polygon2
            print 'polygon3 = ', polygon3
            # # check
            # plt.clf()
            # plt.plot(np.hstack((polygon1[:,0],polygon1[0,0])),np.hstack((polygon1[:,1],polygon1[0,1])),'r-')
            # plt.show()
            # plt.plot(np.hstack((polygon2[:,0],polygon2[0,0])),np.hstack((polygon2[:,1],polygon2[0,1])),'g-')
            # plt.show()
            # plt.plot(np.hstack((polygon3[:,0],polygon3[0,0])),np.hstack((polygon3[:,1],polygon3[0,1])),'b-')
            # plt.show()
            # sys.exit()

        elif shape == 'square':
            ksize = ksize_frac*NH.astype(float)
            ksizeH = ksize
            ksizeV = ksize
            apH = ksizeH*0.5
            apV = ksizeV*0.5
            angleL = apH/np.cos(regalph) # length of rays that come out at angles
            polygon1 = np.array([ [ 0.,0. ], \
                                  [ apH, angleL*np.sin(regalph)], \
                                  [ apH, apV], 
                                  [ 0., apV] ])
            polygon2 = np.array([ [ 0.,0. ], \
                                  [ np.cos(np.pi*0.5+teps), apV] , \
                                  [-apH, apV] , \
                                  [-apH, angleL*np.sin(regbeta)] ])
            polygon3 = np.array([ [ 0.,0. ], \
                                  [-apH, angleL*np.sin(regbeta+teps)], \
                                  [-apH,-apV] , \
                                  [ apH,-apV] , \
                                  [ apH, angleL*np.sin(regalph-teps)] ])

        elif shape == 'circle':
            ksize = float(ksize_frac)*float(NH)
            tt = np.linspace(0.0,2.*np.pi,100)
            print 'tt = ', tt
            print 'np.dstack((np.cos(tt), np.sin(tt))) = ', np.dstack((np.cos(tt), np.sin(tt)))
            print 'ksize = ', ksize
            circlepoly = ksize * np.dstack((np.cos(tt), np.sin(tt)))[0]
            polygon1, polygon2, polygon3 = polyfns.slice_polygon_regions(circlepoly, reggamma, regbeta, regalph)

        else:
            RuntimeError('This shape is not yet supported!')


        if args.polyT:
            polygon1 = np.fliplr(polygon1)
            polygon2_tmp = np.fliplr(polygon3)
            polygon3 = np.fliplr(polygon2)
            polygon2 = polygon2_tmp
            print 'polygon1 = ', polygon1

        if args.poly_offset != 'none':
            if '/' in args.poly_offset:
                splitpo = args.poly_offset.split('/')
            else:
                splitpo = args.poly_offset.split('_')
            print 'split_po = ', splitpo
            poly_offset = np.array([ float(splitpo[0]), float(splitpo[1]) ])
            polygon1 += poly_offset
            polygon2 += poly_offset
            polygon3 += poly_offset

        # Save the previous reg1,2,3
        r1old = reg1
        r2old = reg2
        r3old = reg3
        r1xyold = reg1_xy
        r2xyold = reg2_xy
        r3xyold = reg3_xy

        reg1_xy = le.inds_in_polygon(xy+epskick, polygon1)
        reg2_xy = le.inds_in_polygon(xy+epskick, polygon2)
        reg3_xy = le.inds_in_polygon(xy+epskick, polygon3)

        if args.basis == 'XY':
            reg1 = np.sort(np.vstack((2*reg1_xy,2*reg1_xy+1)).ravel())
            reg2 = np.sort(np.vstack((2*reg2_xy,2*reg2_xy+1)).ravel())
            reg3 = np.sort(np.vstack((2*reg3_xy,2*reg3_xy+1)).ravel())
            r1xy_todo = np.setdiff1d(reg1_xy, r1xyold)
            r2xy_todo = np.setdiff1d(reg2_xy, r2xyold)
            r3xy_todo = np.setdiff1d(reg3_xy, r3xyold)
        elif args.basis == 'psi' :
            print 'stacking regions with right-moving selves...'
            reg1 = np.sort(np.vstack((reg1,NP+reg1)).ravel())
            reg2 = np.sort(np.vstack((reg2,NP+reg2)).ravel())
            reg3 = np.sort(np.vstack((reg3,NP+reg3)).ravel())

        # Add onto the previous nu already computed from previous sum iff reg1_old in reg1, etc
        # First check that new reg1,2,3 contain ALL of elements in reg1_old, reg2_old, reg3_old
        r1ok = len(np.setdiff1d(r1old, reg1)) == 0
        r2ok = len(np.setdiff1d(r2old, reg2)) == 0
        r3ok = len(np.setdiff1d(r3old, reg3)) == 0
        control = True
        print 'regions are ok = ', r1ok and r2ok and r3ok
        if r1ok and r2ok and r3ok and control:
            print 'Continuing sum from last iteration...'
            reg1star = np.setdiff1d(reg1, r1old)
            reg2star = np.setdiff1d(reg2, r2old)
            reg3star = np.setdiff1d(reg3, r3old)
            nusum_cont = True
        else:
            print 'Restarting sum from scratch...'
            reg1star = reg1
            reg2star = reg2
            reg3star = reg3
            nu = 0. + 0. * 1j
            nusum_cont = False

        print 'Summing up h values...'
        dmyi = 0
        for j in reg1star:
            if dmyi % 50 == 0: print 'sum: '+str(dmyi) + '/'+str(len(reg1star))
            for k in reg2:
                for l in reg3:
                    # print 'jkl = ', j, ',', k, ',', l
                    # nu += h[j,k,l]
                    nu += 12*np.pi*1j*(P[j,k]*P[k,l]*P[l,j] - P[j,l]*P[l,k]*P[k,j])
            dmyi +=1

        # Do Astar - Bstar
        if nusum_cont:
            dmyi = 0
            for j in r1old:
                if dmyi % 50 == 0: print 'aux1 sum: ' + str(dmyi) + '/' + str(len(r1old))
                for k in reg2star:
                    for l in reg3:
                        nu += 12 * np.pi * 1j * (P[j, k] * P[k, l] * P[l, j] - P[j, l] * P[l, k] * P[k, j])
                dmyi += 1

            dmyi = 0
            for j in r1old:
                if dmyi % 50 == 0: print 'aux2 sum: ' + str(dmyi) + '/' + str(len(r1old))
                for k in r2old:
                    for l in reg3star:
                        nu += 12 * np.pi * 1j * (P[j, k] * P[k, l] * P[l, j] - P[j, l] * P[l, k] * P[k, j])
                dmyi += 1


        print 'nu = ', nu
        nuV[jj*len(ksize_arr)+kk] = nu
        N_V[jj*len(ksize_arr)+kk] = N
        sysV[jj*len(ksize_arr)+kk] = NP
        Nreg1V[jj*len(ksize_arr)+kk] = len(reg1)
        ksize_V[jj*len(ksize_arr)+kk] = ksize_frac
        ksys_sizeV[jj*len(ksize_arr)+kk] = len(reg1)+len(reg2)+len(reg3)

        # Save regions
        regions = { 'reg1' : reg1, 'reg2' : reg2, 'reg3' : reg3,
                    'polygon1': polygon1, 'polygon2': polygon2, 'polygon3': polygon3,
                    'reg1_xy': reg1_xy, 'reg2_xy': reg2_xy, 'reg3_xy': reg3_xy
                    }
        fn = dataNdir+'params_regs_ksize{0:0.3f}'.format(ksize_frac)+'.txt'
        header = 'Region 1 indices for lattice division, ksize=('+ \
                          '{0:0.5f}'.format(ksize_frac*NH.astype(float)) + ', ' + \
                          '{0:0.5f}'.format(ksize_frac*NV.astype(float)) + ') ' + shape
        le.save_dict(regions, fn, header=header)

        if save_ims:
            plt.clf()
            ax = le.display_lattice_2D(xy, BL, PVxydict=PVxydict, NL=NL, KL=KL,
                                       bs='none', close=False, colorz=False, colormap='BlueBlackRed',
                                       bgcolor='#FFFFFF', axis_off=True)
            ax.scatter(xy[reg1_xy,0], xy[reg1_xy,1], c='r', s=80, marker = 'o', alpha=0.3, label='reg1',zorder=100)
            ax.scatter(xy[reg2_xy,0], xy[reg2_xy,1], c='g', s=80, marker = '^', alpha=0.3, label='reg2',zorder=101)
            ax.scatter(xy[reg3_xy,0], xy[reg3_xy,1], c='b', s=80, marker = 's', alpha=0.3, label='reg3',zorder=102)
            patchList = []
            patchList.append(patches.Polygon(polygon1, color='r'))
            patchList.append(patches.Polygon(polygon2, color='g'))
            patchList.append(patches.Polygon(polygon3, color='b'))
            p = PatchCollection(patchList, cmap=cm.jet, alpha=0.2,zorder=99)
            colors = np.linspace(0, 1, 3)[::-1]
            p.set_array(np.array(colors))
            ax.add_collection(p)
            plt.legend()
            titlestr = r'Division of lattice: $\nu = ${0:0.3f}'.format(nu.real)
            plt.title(titlestr)
            plt.axis('equal')
            plt.savefig(dataNdir+'division_lattice_regions_ksize{0:0.3f}'.format(ksize_frac)+'.png')
            if check:
                plt.show()
    
##########################
# Plot and save result
##########################
nulims = [min(0,min(nuV)*1.2), max(0, max(nuV)*1.2)]
plt.clf()
plt.plot(N_V,nuV,'bo-')
plt.title('Chern number versus system size', fontsize=FSFS)
if shape == 'hexagon':
    plt.xlabel('Number of hexagons on a side', fontsize=FSFS)
else:
    plt.xlabel('System width', fontsize=FSFS)
plt.ylabel('Chern number', fontsize=FSFS)
plt.xlim([0,max(Narr)*1.2])
plt.ylim(nulims)
plt.savefig(datadir+'chern_finsize')

plt.clf()
# for ii in range(len(Narr)):
#     inds = np.arange(ii*len(ksize_arr),(ii+1)*len(ksize_arr))
#     print 'inds = ', inds
#     plt.plot(ksize_V[inds],nuV[inds],'o-',label='N='+str(sysV[ii*len(ksize_arr)]))
plt.plot(sysV,nuV,'bo-')
plt.title('Chern number versus system size', fontsize=FSFS)
plt.xlabel('System size (# sites)', fontsize=FSFS)
plt.ylabel('Chern number', fontsize=FSFS)
plt.xlim([0,max(sysV)*1.2])
plt.ylim(nulims)
plt.savefig(datadir+'chern_finsize_Nsites')

plt.clf()
for ii in range(len(Narr)):
    inds = np.arange(ii*len(ksize_arr),(ii+1)*len(ksize_arr))
    print 'inds = ', inds
    plt.plot(ksize_V[inds],nuV[inds],'o-',label='N='+str(sysV[ii*len(ksize_arr)]))
plt.title('Chern number versus system size', fontsize=FSFS)
plt.xlabel('Fraction of system size in sum', fontsize=FSFS)
plt.ylabel('Chern number', fontsize=FSFS)
plt.xlim([0,max(ksize_arr)*1.2])
plt.ylim(nulims)
plt.legend(loc='best')
plt.savefig(datadir+'chern_finsize_ksizeSum')

plt.clf()
for ii in range(len(Narr)):
    inds = np.arange(ii*len(ksize_arr),(ii+1)*len(ksize_arr))
    print 'inds = ', inds
    plt.plot(ksys_sizeV[inds],nuV[inds],'o-',label='N='+str(sysV[ii*len(ksize_arr)]))
plt.title('Chern number versus system size', fontsize=FSFS)
plt.xlabel('Fraction of system size in sum', fontsize=FSFS)
plt.ylabel('Chern number', fontsize=FSFS)
plt.xlim([0,max(ksys_sizeV)*1.2])
plt.ylim(nulims)
plt.legend(loc='best')
plt.savefig(datadir+'chern_finsize_NsitesSum')


print 'ksize_V =', ksize_V
print 'nuV =', nuV
print 'sysV =', sysV
print 'N_V =', N_V


# Save data as txt
XX = np.dstack((N_V,sysV, Nreg1V, ksize_V,ksys_sizeV,nuV.real))[0]
header = 'N, Nparticles, Nreg1, ksize_frac, ksys_size (note this is 2*NP_summed), nu for Chern calculation: basis='+str(args.basis)+' Omg='+str(args.Omg)+' Omk='+str(args.Omk)
np.savetxt(datadir+'chern_finsize.txt', XX, delimiter= ',', header=header)

# Save parameters as param file
params = {'N' : args.N, 'basis': args.basis, 'Omg': args.Omg, 'Omk': args.Omk, 'ksize_frac': args.ksize_frac,
          'omegac': omegac, 'poly_offset': poly_offset, 'polyT': args.polyT}
le.save_dict(params,datadir+'parameters.txt','Parameters for finite size effect chern number calcs')
