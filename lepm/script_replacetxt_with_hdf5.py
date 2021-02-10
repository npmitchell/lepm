import numpy as np
import lepm.dataio as dio
import h5py
import glob
import subprocess

"""Go through specified directories and dump all txt files of pinning disorder, etc into an hdf5 file"""

parent_dir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/networks/hexagonal/'
subdir = 'hexagonal_square_periodicBC_delta0p667_phi0p000_000031_x_000031*'

subdirs = dio.find_subdirs(subdir, parent_dir)
ellipses = ''
doomg = False
dopin = True

if doomg:
    # Go through all matching subdirs, collect
    for subd in subdirs:
        subd = dio.prepdir(subd)
        omgpins = glob.glob(subd + 'Omg_mean*.txt')

        # dump these into a hdf5
        h5fn = subd + 'omg_configs.hdf5'
        if glob.glob(h5fn):
            rw = "r+"
        else:
            rw = "w"

        with h5py.File(h5fn, rw) as fi:
            keys = fi.keys()
            # print 'keys = ', keys
            for omgpin in omgpins:
                pinname = omgpin.split('/')[-1].split('.')[0]  # .lower()

                # is this pinning configuration already in the hdf5 file?
                if pinname not in keys:
                    print ellipses
                    ellipses = ''
                    print 'pinname = ', pinname
                    # load the pinning
                    pinning = np.loadtxt(subd + pinname + '.txt')

                    # add it to the hdf5 file
                    fi.create_dataset(pinname, shape=np.shape(pinning), data=pinning, dtype='float')
                    # fi.require_dataset(pinname, shape=np.shape(pinning), data=pinning, dtype='f')
                else:
                    ellipses += '.'
                    # print '-> already in hdf5 file: ', pinname

        print ellipses
        # Delete the txt files that we found
        call_list = ['rm'] + omgpins
        print 'call_list = ', call_list[0:3], ' ...'
        # Are there files to delete?
        if len(omgpins) > 0:
            print 'deleting #omgpins.txt = ', len(omgpins)
            # Are there too many files to delete at once?
            if len(omgpins) > 100:
                ntimes = int(np.ceil(len(call_list) / 100)) + 1
                for n in np.arange(ntimes):
                    call_list = ['rm'] + omgpins[n * 100:n * 100 + 100]
                    print 'n * 100:n * 100 + 100] ->', n * 100, ':', n * 100 + 100
                    subprocess.call(call_list)
            else:
                subprocess.call(call_list)

    print 'saved to: ' + subd + 'omg_configs.hdf5'
    # fi = h5py.File(subd + 'omg_configs.hdf5', "r")
    # print '\n\n\n\n\n\n\nfi.keys() = ', fi.keys()
    # fi.close()

    # import h5py
    # parent_dir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/networks/hexagonal/'
    # subdir = 'hexagonal_square_periodicBC_delta0p667_phi0p000_000003_x_000003*'
    # subdirs = dio.find_subdirs(subdir, parent_dir)
    # subd = subdirs[0]
    # subd = dio.prepdir(subd)
    # fi = h5py.File('/Users/npmitchell/Dropbox/Soft_Matter/GPU/networks/hexagonal/' +
    #                'hexagonal_square_periodicBC_delta0p667_phi0p000_000003_x_000003/omg_configs.hdf5', 'r')
    with h5py.File(subd + 'omg_configs.hdf5', "r") as fi:
        print 'len(fi.keys()) = ', len(fi.keys())

if dopin:
    # # Go through all matching subdirs, collect instances of pinpin, replace with pin
    # for subd in subdirs:
    #     subd = dio.prepdir(subd)
    #     pinconfs = glob.glob(subd + 'pinpin_haldane*.txt')
    #     for pinconf in pinconfs:
    #         # pinvals = np.loadtxt(omgpin)
    #         outfn = pinconf.replace('pinpin', 'pin')
    #         subprocess.call(['mv', pinconf, outfn])

    # Go through all matching subdirs, collect
    for subd in subdirs:
        subd = dio.prepdir(subd)
        pinconfs = glob.glob(subd + 'pin_haldane*.txt')

        # dump these into a hdf5
        h5fn = subd + 'pin_configs.hdf5'
        if glob.glob(h5fn):
            rw = "r+"
        else:
            rw = "w"

        with h5py.File(h5fn, rw) as fi:
            keys = fi.keys()
            # print 'keys = ', keys
            for pinconf in pinconfs:
                pinname = pinconf.split('/')[-1].split('.')[0]  # .lower()

                # is this pinning configuration already in the hdf5 file?
                if pinname not in keys:
                    print ellipses
                    ellipses = ''
                    print 'pinname = ', pinname
                    # load the pinning
                    pinning = np.loadtxt(subd + pinname + '.txt')

                    # add it to the hdf5 file
                    fi.create_dataset(pinname, shape=np.shape(pinning), data=pinning, dtype='float')
                    # fi.require_dataset(pinname, shape=np.shape(pinning), data=pinning, dtype='f')
                else:
                    ellipses += '.'
                    # print '-> already in hdf5 file: ', pinname

        print ellipses
        # Delete the txt files that we found
        call_list = ['rm'] + pinconfs
        print 'call_list = ', call_list[0:3], ' ...'
        # Are there files to delete?
        if len(omgpins) > 0:
            print 'deleting #omgpins.txt = ', len(pinconfs)
            # Are there too many files to delete at once?
            if len(pinconfs) > 100:
                ntimes = int(np.ceil(len(call_list) / 100)) + 1
                for n in np.arange(ntimes):
                    call_list = ['rm'] + pinconfs[n * 100:n * 100 + 100]
                    print 'n * 100:n * 100 + 100] ->', n * 100, ':', n * 100 + 100
                    subprocess.call(call_list)
            else:
                subprocess.call(call_list)

    print 'saved to: ' + subd + 'pin_configs.hdf5'
    # fi = h5py.File(subd + 'omg_configs.hdf5', "r")
    # print '\n\n\n\n\n\n\nfi.keys() = ', fi.keys()
    # fi.close()

    with h5py.File(subd + 'pin_configs.hdf5', "r") as fi:
        print 'len(fi.keys()) = ', len(fi.keys())
