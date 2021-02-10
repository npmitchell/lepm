import h5py
import numpy as np
import glob
import lepm.lattice_elasticity as le
import argparse
import cPickle as pkl

"""Copy data from chern calculations into hdf5 files on local disk: ie, pull from research4TB onto local disk.
"""

parser = argparse.ArgumentParser(description='Copy data from chern calculations into hdf5 files on local disk.')
parser.add_argument('-outdir', '--outdir', help='Path to data to be output in hdf5 format',
                    type=str, default="/Users/npmitchell/Dropbox/Soft_Matter/GPU/kitaev_chern/")
parser.add_argument('-datadir', '--datadir', help='Path to data in directory-and-txt-file format',
                    type=str, default="/Volumes/research4TB/Soft_Matter/GPU/chern/")
parser.add_argument('-overwrite', '--overwrite_existing_data', help='', action='store_true')
parser.add_argument('-verbose', '--verbose', help='', action='store_true')
args = parser.parse_args()


def load_chern_hdf5(cherndir, cn, overwrite=False, verbose=False):
    """
    From cherndir, load chern_finsize.txt, chern_params.txt, lattice_params.txt, params_regs, and contribs into
    hdf5 group cn.

    Parameters
    ----------
    cherndir : str
        Path to chern data on disk
    cn : h5py subgroup
        Where the data will be stored in hdf5 format
    overwrite : bool
        Overwrite the data if already exists in the hdf5 format
    verbose : bool
        Output more to command line
    """
    # load numpy array chern_finsize.txt
    print 'loading ' + le.prepdir(cherndir) + 'chern_finsize.txt'
    chern_finsize = np.loadtxt(le.prepdir(cherndir) + 'chern_finsize.txt', delimiter=',')
    cn.attrs['chern_finsize'] = chern_finsize

    # load dict chern_params.txt
    cp = le.load_params(le.prepdir(cherndir) + 'chern_params.txt')
    if 'cp' not in cn:
        cpgroup = cn.create_group('cp')
        # By continuing we are not overwriting data
        overwriting = False
    else:
        cpgroup = cn['cp']
        # By continuing we are overwriting data
        overwriting = True

    if overwrite or not overwriting:
        for key in cp:
            cpgroup.attrs[key] = cp[key]

        # load lattice_params.txt if it exists
        if glob.glob(cherndir + 'lattice_params.txt'):
            # load dict chern_params.txt
            if verbose:
                print 'creating lp group in ', cn
            lp = le.load_params(le.prepdir(cherndir) + 'lattice_params.txt')
            if 'lp' not in cn:
                lpgroup = cn.create_group('lp')
            else:
                lpgroup = cn['lp']
            for key in lp:
                lpgroup.attrs[key] = lp[key]

        # Store params_regs
        if glob.glob(cherndir + 'params_regs/'):
            pregs = cn.create_group("params_regs")

            for preg_fn in glob.glob(cherndir + 'params_regs/params_regs_*.txt'):
                pregname = preg_fn.split('/')[-1]
                # load dict params_regs_ksize#p###.txt
                params_regs = le.load_params(preg_fn)
                prgroup = pregs.create_group(pregname)
                for key in params_regs:
                    prgroup.attrs[key] = params_regs[key]
        elif glob.glob(cherndir + 'params_regs.pkl'):
            if verbose:
                print 'found params_regs pickle: ' + cherndir + 'params_regs.pkl'
            with open(glob.glob(cherndir + 'params_regs.pkl')[0], "rb") as fn:
                pregdict = pkl.load(fn)
            if "params_regs" not in cn:
                pregs = cn.create_group("params_regs")
            else:
                pregs = cn["params_regs"]
            # print 'pregdict = ', pregdict
            for prkey in pregdict:
                if verbose:
                    print 'Creating group in pregs: ', prkey
                params_regs = pregdict[prkey]
                prgroup = pregs.create_group(prkey)
                for key in params_regs:
                    prgroup.attrs[key] = params_regs[key]

        # Store contribs
        if glob.glob(cherndir + 'contribs/'):
            print 'found contributions directory: ' + cherndir + 'contribs/'
            cons = cn.create_group("contribs")
            for contrib_fn in glob.glob(cherndir + 'contribs/contribs_*.txt'):
                contribname = contrib_fn.split('/')[-1]
                # load dict contribs_ksize#p###.txt
                contribs = le.load_params(contrib_fn)
                print 'contribname = ', contribname
                ctgroupname = contribname.split('_')[-1].split('.')[0].split(['ksize'])[-1]
                print 'ctgroupname = ', ctgroupname
                ctgroup = cons.create_group(ctgroupname)
                for key in cons:
                    ctgroup.attrs[key] = contribs[key]
        elif glob.glob(cherndir + 'contribs.pkl'):
            print 'found contributions pickle: ' + cherndir + 'contribs.pkl'
            with open(glob.glob(cherndir + 'contribs.pkl')[0], "rb") as fn:
                contribs = pkl.load(fn)
            if 'contribs' not in cn:
                cons = cn.create_group("contribs")
            else:
                cons = cn['contribs']
            for ckey in contribs:
                if verbose:
                    print 'Creating group in cons: ', ckey
                contrib = contribs[ckey]
                ctgroup = cons.create_group(ckey)
                for key in contrib:
                    ctgroup.attrs[key] = contrib[key]
    else:
        print 'Data already exists and overwrite=False: skipping load...'

    return cn


outdir = args.outdir
datadir = args.datadir

# Each meshfn has subgroups for each chern calc.
# Each chern calc has key value pairs for
#         chern.cp
#         chern.chern_finsize = None
#         chern.contribs = None
#         chern.params_regs = {}

# tmp = ['/Volumes/research4TB/Soft_Matter/GPU/chern/hexagonal']


# Example walkthough attributes:
# def print_attrs(name, obj):
#         print name
#         for key, val in obj.attrs.iteritems():
#             print "    %s: %s" % (key, val)
#
# f.visititems(print_attrs)

for ltdir in glob.glob(datadir + '*'):
    lt = ltdir.split('/')[-1]
    print 'lt = ', lt
    chernfn = outdir + lt + '.hdf5'
    if glob.glob(chernfn):
        f = h5py.File(chernfn, "r+")
        print 'opened', chernfn, 'for reading/writing'
    else:
        f = h5py.File(chernfn, "w")
        print 'opened ', chernfn, ' for writing'

    # Now look inside ltdir for subdirs which are meshfns
    # Get meshfns which are dirs
    meshfns = le.find_subdirs(lt + '*', ltdir)
    print 'meshfns = ', meshfns
    for meshfndir in meshfns:
        meshfn = meshfndir.split('/')[-2]
        print 'meshfn = ', meshfn

        if meshfn not in f:
            fmf = f.create_group(meshfn)
        else:
            fmf = f[meshfn]

        # Get chernfns which are dirs
        chernfns = le.find_subdirs('Omk*polyoff*', meshfndir)
        for cherndir in chernfns:
            chernname = cherndir.split('/')[-2]

            # Check if chernname holds the chern calculation or is a set of chern calculations.
            # If chernname ends in _XY, then it holds the chern calculation. Otherwise, it is a set.
            if cherndir[-4:-1] == '_XY':
                # If chern subgroup in meshfn subgroup has not been made and data exists, load it
                if chernname not in fmf and glob.glob(le.prepdir(cherndir) + 'chern_finsize.txt'):
                    cn = fmf.create_group(chernname)
                    load_chern_hdf5(cherndir, cn, overwrite=args.overwrite_existing_data, verbose=args.verbose)
                elif glob.glob(le.prepdir(cherndir) + 'chern_finsize.txt'):
                    cn = fmf[chernname]
                    load_chern_hdf5(cherndir, cn, overwrite=args.overwrite_existing_data, verbose=args.verbose)
                else:
                    print 'data missing, skip chern...'
            else:
                # Chernname is a set of calculations at different positions in the network.
                if chernname not in fmf:
                    csubgroup = fmf.create_group(chernname)
                else:
                    csubgroup = fmf[chernname]

                # Get chernfns which are dirs: offset in Y is a subdir
                yoff_fns = le.find_subdirs('*_XY', cherndir)
                for yoffdir in yoff_fns:
                    yoff = yoffdir.split('/')[-2]
                    if yoff not in csubgroup and glob.glob(le.prepdir(yoffdir) + 'chern_finsize.txt'):
                        cn = csubgroup.create_group(yoff)
                        load_chern_hdf5(yoffdir, cn, overwrite=args.overwrite_existing_data, verbose=args.verbose)
                    elif glob.glob(le.prepdir(yoffdir) + 'chern_finsize.txt'):
                        load_chern_hdf5(yoffdir, cn, overwrite=args.overwrite_existing_data, verbose=args.verbose)
                    else:
                        print 'data missing, skip chern...'

    f.close()
