import h5py
import glob
import numpy as np

'''Auxiliary functions for reading and writing to hdf5 files
'''


def dset_in_hdf5(dset_name, hdf5fn):
    """Check if a dataset with a given name dset_name exists in the hdf5 file whose path is hdf5fn

    Parameters
    ----------
    dset_name : str
        The key of the dataset to look for in the specified hdf5 file (no subgroup)
    hdf5fn : str
        The path and filename of the hdf5 file to look in

    Returns
    -------
    inhdf : bool
        The dataset with name dset_name is in the hdf5 file
    """
    if glob.glob(hdf5fn):
        with h5py.File(hdf5fn, 'r+') as fi:
            # is this pinning configuration already in the hdf5 file?
            # print 'hdf5io: fi.keys() = ', fi.keys()
            # print 'hdf5io: dset_name = ', dset_name
            if dset_name in fi.keys():
                return True
            else:
                return False
    else:
        return False


def extract_dset_hdf5(dset_name, hdf5fn):
    """

    Parameters
    ----------
    dset_name : str
    hdf5fn : str

    Returns
    -------
    dset or None
    """
    if glob.glob(hdf5fn):
        print 'hdf5io: dset exists, trying to find ' + dset_name
        with h5py.File(hdf5fn, 'r+') as fi:
            if dset_name in fi.keys():
                print 'hdf5io: dset is in fi.keys(): ' + dset_name
                dset = fi[dset_name][:]
                return dset
            else:
                return None
    else:
        return None


def save_dset_hdf5(dset, dset_name, hdf5fn, overwrite=False, dtype=None):
    """

    Parameters
    ----------
    dset_name
    hdf5fn
    overwrite : bool
        Overwrite the dataset even if it exists in the file already

    Returns
    -------

    """
    # Obtain datatype for this dataset
    if dtype is None:
        dtype = dset.dtype

    # Determine whether we are creating the hdf5 file or just adding to it
    if glob.glob(hdf5fn):
        rw = "r+"
    else:
        rw = "w"

    # Store the dataset in the hdf5 file
    with h5py.File(hdf5fn, rw) as fi:
        # is this dset already in the hdf5 file?
        if dset_name not in fi.keys() or overwrite:
            # add dset to the hdf5 file
            # print 'hdf5io: saving dset_name = ', dset_name
            # print 'hdf5io: to hdf5fn = ', hdf5fn
            # try:
            fi.create_dataset(dset_name, shape=np.shape(dset), data=dset, dtype=dtype)
            # except RuntimeError:
            #     print '\n\nhdf5io: Name already exists! Removing and overwriting...'
            #     print 'hdf5io: dset_name = ', dset_name
            #     print 'hdf5io: hdf5fn = ', hdf5fn
            #     dset = extract_dset_hdf5(dset_name, hdf5fn)
            #     print 'hdf5io: dset = ', dset
            #     sys.exit()
        else:
            print 'dset_name = ', dset_name
            print 'hdf5fn = ', hdf5fn
            raise RuntimeError('dset already exists in hdf5, exiting...')
