import numpy as np
import numpy.linalg as la
import lepm.lattice_elasticity as le
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
try:
    import cPickle as pickle
except ImportError:
    import pickle
import glob
import lattice_class
import gyro_lattice_class
import plotting.plotting as leplt

'''
Calculate the diffisivity d(omega) for a harmonic network (masses and springs).
'''


def delta_eo(eta, omega, omegaj):
    """delta_eta(omega)"""
    return eta / (np.pi * (omega - omegaj)**2 + eta**2)


parser = argparse.ArgumentParser(description='Specify parameters for computing diffusivity of harmonic network eigenmodes.')
parser.add_argument('datehourmin', type=str, nargs='?',
                    help='Name of simulation to build, in format YYYYMMDD-HHMM',
                    default='check_string_for_empty')
parser.add_argument('-seriesdir', '--seriesdir',
                    help='Name for the directory in which to store the calculation datadir', type=str,
                    default='diffusivity_harmonic')
parser.add_argument('-basis', '--basis' ,help='Basis in which to compute the eigenvectors, projector, etc',
                    type=str, default='XY')
parser.add_argument('-periodic', '--periodicBC', help='Enforce periodic boundary conditions', action='store_true')
parser.add_argument('-skip_ims', '--skip_ims', help='Do not save DOS or region images', action='store_true')
parser.add_argument('-check', '--check', help='Show progress images during calculation', action='store_true')
parser.add_argument('-checkout', '--checkout', help='Save progress images during calculation', action='store_true')
parser.add_argument('-Omg', '--Omg', help='Frequency of gravitational precession (specify if homogenous across sample)',
                    type=float, default=1.0)
parser.add_argument('-Omk', '--Omk', help='Frequency of spring interaction (specify if homogenous across sample)',
                    type=float, default=1.0)
parser.add_argument('-ksize', '--ksize_frac',
                    help='Fraction of system size to include in kitaev calculation. Specify a series of values if desired',
                    type=str, default='0.75')
parser.add_argument('-skip_DOS_ims', '--skip_DOS_ims', help='Skip plotting and saving Density Of States results',
                    action='store_true')
parser.add_argument('-save_DOS', '--save_DOS', help='Save Density Of States results as pickle files',
                    action='store_true')
parser.add_argument('-N', '--N',
                    help='Mesh width AND height, in number of lattice spacings. Specify a series of values by using / between vals (ex 3/4/5/6)',
                    type=str, default='5')

# Geometry
parser.add_argument('-slit', '--slit', help='Declare whether or not there is a slit', action='store_true')
parser.add_argument('-NH', '--NH', help='Mesh width, in number of lattice spacings', type=int, default=10)
parser.add_argument('-NV', '--NV', help='Mesh height, in number of lattice spacings', type=int, default=10)
parser.add_argument('-NP', '--NP_load',
                    help='System size for entire loaded network (by default, with PBCs). Overrides NH,NV.'+\
                    'Specify a series of values by using / between vals (ex 3/4/5/6)',
                    type=str, default='0')
parser.add_argument('-LT', '--LatticeTop', \
                    help='Lattice topology: linear, hexagonal, triangular, deformed_kagome, twisted_kagome, square, hyperuniform, hyperuniformdual, isostatic', \
                    type=str, default='jammed')
parser.add_argument('-delta', '--delta_lattice', help='for hexagonal lattice, measure of deformation in radians/pi',
                    type=str, default='0.667')
parser.add_argument('-phi', '--phi_lattice', help='for hexagonal lattice, measure of deformation in radians/pi',
                    type=str, default='0.000')
parser.add_argument('-eta', '--eta', help='Lattice randomization/jitter (usually units of unity/lattice spacing)',
                    type=float, default=0.000)
parser.add_argument('-theta', '--theta', help='Lattice rotation (units of pi radians)', type=float, default=0.000)
parser.add_argument('-x1', '--x1', help='Deformed kagome geometric parameter', type=float, default=0.000)
parser.add_argument('-x2', '--x2', help='Deformed kagome geometric parameter', type=float, default=0.000)
parser.add_argument('-x3', '--x3', help='Deformed kagome geometric parameter', type=float, default=0.000)
parser.add_argument('-zkagome', '--zkagome', help='Deformed kagome geometric parameter', type=float, default=0.000)
parser.add_argument('-cutL', '--cutL', help='String specifying length of slit as fraction of L', type=str,
                    default='0.5')
parser.add_argument('-shape', '--shape', help='Shape of the overall mesh geometry', type=str, default='square')
parser.add_argument('-huID', '--hyperuniform_number',
                    help='If LT=hyperuniform, selects which hyperuniform lattice to use', type=str, default='01')
parser.add_argument('-z', '--target_z',
                    help='Average coordination of the lattice to load. If -1, loads default coord', type=str,
                    default='-1.')
parser.add_argument('-cutz_method', '--cutz_method',
                    help='random or highest, method by which z target value is obtained', type=str, default='none')
parser.add_argument('-Ndefects', '--Ndefects', help='Number of defects to introduce', type=int, default=1)
parser.add_argument('-Bvec', '--Bvec', help='Direction of burgers vectors of dislocations (random, W, SW, etc)',
                    type=str, default='W')
parser.add_argument('-dislocxy', '--dislocation_xy',
                    help='Position of single dislocation, if not centered at (0,0), as strings sep by / (ex: 1/4.4)',
                    type=str, default='none')
parser.add_argument('-deform', '--deform', help='Whether to deform the lattice over time in sim', action='store_true')
parser.add_argument('-deform_rate', '--deform_rate', help='Rate in radians/unit time to deform lattice', type=float,
                    default=0.001)
parser.add_argument('-adiabatic', '--deform_adiabatic', help='Whether to drag xy along with xy0 in deformation',
                    action='store_true')
parser.add_argument('-alph', '--tk_alph', help='Twisted kagome twist angle, in radians', type=float, default=0.000)
parser.add_argument('-alph_final', '--alph_final',
                    help='Final deformation parameter in realtime deform (for twisted_kagome, this is pi/3)',
                    type=float, default=1.0471975511965976)
args = parser.parse_args()

maindir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/experiments/'
# Get timestamp for this calculation
print '\n args.datehourmin = ', args.datehourmin
timestr, newdatadir, LOADPARAMS, datadir = le.determine_simulation_timestamp(args.datehourmin,maindir)
datadir = maindir+args.seriesdir+'/'+args.LatticeTop
if args.delta_lattice != '0.667':
    datadir += '_delta' + args.delta_lattice.replace('.','p')

datadir += '/'

# Series of sizes to run is N array (Narr) or series of other values
if args.NP_load == 0:
    Narr = le.string_sequence_to_numpy_array(args.N, dtype = int)
    doNP = False
else:
    Narr = le.string_sequence_to_numpy_array(args.NP_load, dtype = int)
    doNP = True

ksize_arr = le.string_sequence_to_numpy_array(args.ksize_frac, dtype = float)
print 'Narr = ', Narr
print 'type(Narr) = ', type(Narr)
sysV = np.zeros(len(Narr)*len(ksize_arr), dtype=int)
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


# Use each value of N
for kk in range(len(Narr)):
    if doNP:
        N = Narr[kk]
    else:
        N = Narr[kk]

    print '\nN = ', N
    dataNdir = datadir+'N_{0:03d}'.format(N)+'/'
    le.ensure_dir(dataNdir)
    NH = N
    NV = N
    Ns = min(NH, NV)

    # Get polygon over which to sum
    if LatticeTop in ['triangular', 'triangularz']:
        dist = 1.
    # else:
    #    # distance for NH=1 is dist
    #    if args.delta == '0.667':
    #        dist = 2.*le.polygon_apothem(1.0,6) # distance across one hexagon of bonds with bondlength=1
    #    else:
    #        a1, a2 = deformed_hexcell_to_hexagonal_sidelengths(args.delta,args.NH,args.NV)

    # Make Lattice
    sourcedir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU'
    eta = args.eta
    if args.slit:
        cutLstr = '_cutL'+args.cutL+'L_x0_*'  # '_cutL0.2L_x0_*' ''\
    else:
        cutLstr = ''
    lp = { 'x1' : args.x1,
        'x2' : args.x2,
        'x3' : args.x3,
        'zkagome' : args.zkagome,
        'alph' : args.tk_alph,
        'periodicBC' : args.periodicBC,
        'cutLstr' : cutLstr,
        'delta_lattice' : args.delta_lattice,
        'phi_lattice' : args.phi_lattice,
        'theta_lattice' : args.theta,
        'eta' : args.eta,
        'huID' : args.hyperuniform_number,
        'z' : args.target_z,
        'Ndefects' : args.Ndefects,
        'Bvec' : args.Bvec,
        'dislocxy' : (args.dislocation_xy.split('/')),
        'cutz_method' : args.cutz_method,
        'origin' : np.array([0.,0.]),
        'source': 'hexner',
        'Omk': -1.0,
        'Omg': -1.0,
    }
    if doNP:
        print 'check'
        lp['NP_load'] = N
    else:
        lp['NP_load'] = 0

    meshfn = le.find_meshfn(args.LatticeTop, shape, sourcedir, NH, NV, lattice_params = lp)
    print 'found meshfn = ', meshfn

    if glob.glob(meshfn+'/diffusivity_sigmax.pkl'):
        with open(meshfn+'/diffusivity_sigma.pkl', "rb") as input_file:
            SS = pickle.load(input_file)
        with open(meshfn+'/diffusivity_sigmax.pkl', "rb") as input_file:
            SSx = pickle.load(input_file)
        with open(meshfn+'/diffusivity_sigmay.pkl', "rb") as input_file:
            SSy = pickle.load(input_file)
        with open(meshfn+'/eigval_mass.pkl', "rb") as input_file:
            eigval = pickle.load(input_file)
    else:
        # Load lattice
        lat = lattice_class.lattice()
        lat.load(meshfn)
        glat = gyro_lattice_class.gyro_lattice(lat, lp)
        glat.load(meshfn=meshfn)
        xy = glat.lattice.xy

        # Make or load Dynamical Matrix
        if args.basis == 'XY':
            # Check if eigvect, eigval, DM exist
            print '/n/n/n find-->', glob.glob(meshfn + '/eigval_mass.pkl')
            eigvalfn = glob.glob(meshfn + '/eigval_mass.pkl')
            eigvectfn = glob.glob(meshfn + '/eigvect_mass.pkl')
            DMfn = glob.glob(meshfn + '/dynamical_matrix_mass.pkl')
            print 'eigvalfn = ', eigvalfn
            if eigvalfn and eigvectfn and DMfn:
                print '\n\n found eigval, eigvect. Loading...'
                with open(eigvalfn[0], "rb") as input_file:
                    eigval = pickle.load(input_file)
                with open(eigvectfn[0], "rb") as input_file:
                    eigvect = pickle.load(input_file)
                with open(DMfn[0], "rb") as input_file:
                    HH = pickle.load(input_file)
                # fig, DOS_ax = leplt.initialize_DOS_plot(eigval*1j, 'gyro', pin=- 5000)
                # plt.show()
            else:
                print '\n\nDid not find all of eigval, eigvect, dynamical matrix. Computing...'
                DD = glat.calc_matrix()
                glat.eigval, glat.eigvect = glat.eig_vals_vects()
                glat.plot_eigval_hist(infodir=meshfn, show=True)
                eigval = glat.eigval
                eigvect = glat.eigvect
                print '\n\n Saving computed eigval, eigvect...'
                output = open(meshfn + '/eigval.pkl', 'wb')
                pickle.dump(eigval, output)
                output.close()
                output = open(meshfn + '/eigvect.pkl', 'wb')
                pickle.dump(eigvect, output)
                output.close()
                output = open(meshfn + '/dynamical_matrix.pkl', 'wb')
                pickle.dump(DD, output)
                output.close()
                if not glob.glob(meshfn+'/DOS_gryo.mov'):
                    le.plot_movie_normal_modes_Nashgyro(meshfn+'/', xy, NL, KL, OmK, Omg, sim_type='gyro',
                                                        rm_images=True, save_ims=True, gapims_only=False)

        # Sigma = np.dot(eigvect,np.dot(HH, eigvect))
        # le.plot_complex_matrix(np.log(Sigma), name=r'$\Sigma_{mn}$', climvs=[[-6,3], [-6,3]], show=True)

        # # Test data
        # rvecs = np.array([[0,0],[0,1],[1,0],[1,1]])
        # eigvect = np.array([[1,2,3,4],[5,6,7,8],[9,0,1,2],[3,4,5,6],[7,8,9,10],[11,12,13,14],
        #                     [15,16,17,18],[19,20,21,22]])
        # eigval = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        # HH = np.array([[0,1,0,0],[0,1,0,0],[1,0,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0],[1,1,1,0],[1,1,1,0],
        #                [0, 1, 1, 1], [0, 1, 1, 1],[1,0,1,1],[1,0,1,1],[1,1,0,1],[1,1,0,1],[1,1,1,0],[1,1,1,0],])
        # print np.dot(eigvect,np.dot(HH, eigvect))
        # sys.exit()

        SSx = np.zeros((len(eigval), len(eigval)), dtype=complex)
        SSy = np.zeros((len(eigval), len(eigval)), dtype=complex)
        summn = 0.0
        for mm in range(len(eigval)):
            if mm % 1 == 0: print 'Calculating SS for eigvect ', mm, ' of ', len(eigval)
            for nn in range(len(eigval)):
                if nn % 50 == 0: print nn, ' of ', len(eigval)
                # pref = ((eigval[nn] + eigval[nn]) ** 2) / (4 * eigval[nn] * eigval[nn])
                summn = 0.
                for ii in range(len(xy)):
                    for jj in range(len(xy)):
                        distv = xy[ii] - xy[jj]
                        product00 = eigvect[mm, 2 * ii] * HH[2 * ii, 2 * jj] * eigvect[nn, 2 * jj]
                        product01 = eigvect[mm, 2 * ii] * HH[2 * ii, 2*jj+1] * eigvect[nn, 2*jj+1]
                        product10 = eigvect[mm, 2*ii+1] * HH[2*ii+1, 2 * jj] * eigvect[nn, 2 * jj]
                        product11 = eigvect[mm, 2*ii+1] * HH[2*ii+1, 2*jj+1] * eigvect[nn, 2*jj+1]
                        summn += distv * ( product00 + product01 + product10 + product11 )
                toadd = summn  # pref * summn
                SSx[mm, nn] += toadd[0]
                SSy[mm, nn] += toadd[1]
            # print 'SSx = ', SSx

        SS = np.abs(SSy**2 + SSx ** 2)
        ####################
        # Save data
        ####################
        output = open(datadir+str(N)+'_SS.pkl', 'wb')
        pickle.dump(SS, output)
        output.close()
        output = open(datadir+str(N)+'_SSx.pkl', 'wb')
        pickle.dump(SSx, output)
        output.close()
        output = open(datadir+str(N)+'_SSy.pkl', 'wb')
        pickle.dump(SSy, output)
        output.close()
        output = open(meshfn + '/diffusivity_sigma.pkl', 'wb')
        pickle.dump(SS, output)
        output.close()
        output = open(meshfn + '/diffusivity_sigmax.pkl', 'wb')
        pickle.dump(SSx, output)
        output.close()
        output = open(meshfn + '/diffusivity_sigmay.pkl', 'wb')
        pickle.dump(SSy, output)
        output.close()


    ###################
    # PLOT HEAT FLUX
    ###################
    le.plot_complex_matrix(np.log(SSy), name=r'$\ln\Sigma_y$', climvs=[[-1,2], [-1,2]], show=True)
    le.plot_complex_matrix(np.log(SSx), name=r'$\ln\Sigma_x$', climvs=[[-1,2], [-1,2]], show=True)
    le.plot_real_matrix(SS, name=r'$|\Sigma|^2$', outpath=meshfn+'/diffusivity_sigma', climv=(0,1))
    le.plot_real_matrix(np.log(SS), name=r'$\ln|\Sigma|^2$', outpath=meshfn+'/diffusivity_sigmasquared')
    le.plot_pcolormesh_scalar(eigval, eigval, SS, outpath=meshfn+'/diffusivity_sigma_pcolor', title=r"$|\Sigma(\omega, \omega')|^2$",
                              vmax=.11, vmin=0, xlabel=r'$\omega$', ylabel=r"$\omega'$", show=False, cmap='bone')
    le.plot_pcolormesh_scalar(eigval, eigval, SS, outpath=meshfn+'/diffusivity_sigma_pcolor_wide', title=r"$|\Sigma(\omega, \omega')|^2$",
                              vmax=1, vmin=0, xlabel=r'$\omega$', ylabel=r"$\omega'$", show=False, cmap='bone')
    le.plot_pcolormesh_scalar(eigval, eigval, np.log10(SS), outpath=meshfn+'/diffusivity_sigma_pcolor_log', title=r"$\log_{10} |\Sigma(\omega, \omega')|^2$",
                              vmax=0, vmin=-6, xlabel=r'$\omega$', ylabel=r"$\omega'$", show=False, cmap='bone')

    # Also save in experiments
    plt.savefig(datadir+str(N)+'_SS.png')

    ###################
    # Translate into diffusivity
    ###################
    # Compute delta_eta(omega_i - omega_j)
    omegadiff = le.diff_matrix(eigval, eigval)
    eta = 5.0 * np.mean(np.diff(eigval))

    diffusivity = np.zeros(len(eigval))
    for ii in range(len(eigval)):
        for jj in range(len(eigval)):
            diffusivity[ii] += (eigval[ii] + eigval[jj])**2.0/(4.*eigval[ii]*eigval[jj]) * SS[ii,jj] * delta_eo(eta, eigval[ii], eigval[jj])
        print 'adding up diffusivity: ', ii, ' of ', len(eigval)

    diffusivity *= np.pi/(12. * eigval[ii]**2.0)
    output = open(meshfn + '/diffusivity.pkl', 'wb')
    pickle.dump(diffusivity, output)
    output.close()
    output = open(datadir + str(N) + '_diffusivity.pkl', 'wb')
    pickle.dump(diffusivity, output)
    output.close()

    plt.loglog(eigval, diffusivity, 'b.-')
    plt.title(r'Diffusivity $d(\omega)$')
    #plt.ylim(0,10)
    plt.xlim(0.02,4)
    plt.savefig(meshfn+'/diffusivity.png')
    plt.savefig(datadir+str(N)+'_diffusivity.png')
    plt.show()

