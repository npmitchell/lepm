import numpy as np
import lepm.lattice_elasticity as le
import lepm.kitaev_experiment_functions as kexfns
import glob
import argparse
try:
    import cPickle as pickle
except ImportError:
    import pickle

'''Calculate Chern number via realspace method, as a script, using only eigval, eigvect, and xy.

Example usage:
python kitaev_chern_calc_experimental_script.py -rootdir /Users/npmitchell/Desktop/test/eexy/ -save_ims -shape hexagon -modsave 5 -ksize_frac_array 0.0:0.01:0.50
'''

####################################################################################
# Script
####################################################################################
parser = argparse.ArgumentParser('Compute the chern number from an experimental run')
parser = argparse.ArgumentParser(description='Specify time string (timestr) for gyro simulation.')
parser.add_argument('-rootdir', '--rootdir', help='Root directory for where data is located', type=str,
                    default='/Users/npmitchell/Dropbox/GPU/')
parser.add_argument('-seriesdir', '--seriesdir', help='Output subdirectory of cprootdir', type=str, default='none')
parser.add_argument('-contributions', '--contributions',
                    help='Calculate contributions from each gyro to chern number for an array of kitaev sum sizes',
                    action='store_true')

# chern parameters
parser.add_argument('-ksize_frac_array', '--ksize_frac_array',
                    help='Array of fractional sizes to make the kitaev region, specified with /s', type=str,
                    default='0.0:0.01:1.10')
parser.add_argument('-omegac', '--omegac', help='cutoff frequency for projector', type=float, default=2.25)
parser.add_argument('-regalph', '--regalph', help='largest angle dividing kitaev region',
                    type=float, default=np.pi * (11. / 6.))
parser.add_argument('-regbeta', '--regbeta', help='middle angle dividing kitaev region',
                    type=float, default=np.pi * (7. / 6.))
parser.add_argument('-reggamma', '--reggamma', help='smallest angle dividing kitaev region',
                    type=float, default= np.pi * 0.5)
parser.add_argument('-polyT', '--polyT', help='whether to transpose the kitaev region', action='store_true')
parser.add_argument('-poly_offset', '--poly_offset', help='coordinates to translate the kitaev region, as string',
                    type=str, default='none')
parser.add_argument('-basis', '--basis', help='basis for performing kitaev calculation (XY, psi)',
                    type=str, default='XY')
parser.add_argument('-modsave', '--modsave',
                    help='How often to output an image of the kitaev region and calculation result',
                    type=int, default=20)
parser.add_argument('-save_ims', '--save_ims', help='Whether to save images of the calculations',
                    action='store_true')
parser.add_argument('-shape', '--shape', help='Shape of the overall mesh geometry', type=str, default='square')
parser.add_argument('-check', '--check', help='Check outputs during computation of lattice', action='store_true')
parser.add_argument('-verbose', '--verbose', help='Print more to command line during computation', action='store_true')

# Global geometric params
parser.add_argument('-delta', '--delta', help='for hexagonal kitaev region, deformation opening angle',
                    type=float, default=2./3.)
args = parser.parse_args()

####################################################################################
####################################################################################
cprootdir = '/Users/npmitchell/Desktop/test/chern/'

# Grab and prepare directories for both input (datadir) and output (outdir)
if args.seriesdir is not 'none':
    outdir = le.prepdir(cprootdir + args.seriesdir)
    datadir = le.prepdir(args.rootdir + args.seriesdir)
else:
    outdir = le.prepdir(cprootdir)
    datadir = args.rootdir

le.ensure_dir(outdir)

cp = {'ksize_frac_arr': le.string_sequence_to_numpy_array(args.ksize_frac_array, dtype=float),
      'omegac': args.omegac,
      'regalph': args.regalph,
      'regbeta': args.regbeta,
      'reggamma': args.reggamma,
      'shape': args.shape,
      'polyT': args.polyT,
      'poly_offset': args.poly_offset,
      'basis': args.basis,
      'modsave': args.modsave,
      'save_ims': args.save_ims,
      'rootdir': cprootdir,
      'cpmeshfn': outdir,
      'outerH': 1.0,
}


# Load eigenvals and eigenvects
with open(datadir + 'eigval.pkl', "rb") as input_file:
    eigval = pickle.load(input_file)
with open(datadir + 'eigvect.pkl', "rb") as input_file:
    eigvect = pickle.load(input_file)

xyload = np.loadtxt(datadir + 'xy.txt', delimiter=',', skiprows=1, usecols=(0, 1), unpack=True)
xy0 = (xyload.T).astype(float)
xy = np.dstack((xy0[:, 0], xy0[:, 1]))[0]

chern_finsize, params_regs, contribs = \
    kexfns.calc_kitaev_chern_from_evs(xy, eigval, eigvect, cp, pp=None, check=False, contributions=args.contributions,
                                      verbose=args.verbose, vis_exten='.png', contrib_exten='.pdf', delta=args.delta)

# Save everything
if contribs is not None:
    with open(outdir + 'contribs.pkl', "wb") as fn:
        pickle.dump(contribs, fn)

with open(outdir + 'params_regs.pkl', "wb") as fn:
    pickle.dump(params_regs, fn)

fn = outdir + 'chern_finsize.txt'
header = '# Nreg1, ksize_frac, ksize, ksys_size (note this is 2*NP_summed), ksys_frac, nu for expt Chern calculation'
np.savetxt(fn, chern_finsize, delimiter=',', header=header)

fn = outdir + 'chern_params.txt'
header = '# Parameters for chern calculation'
le.save_dict(cp, fn, header=header)


print 'done!'

