import numpy as np
import lepm.dataio as dio
import lepm.stringformat as sf
import lepm.math_functions as amath
from lepm.lattice_elasticity import eig_vals_vects
import lepm.magnetic_gyro_kspace_functions as mglatkfns
import lepm.data_handling as dh
import lepm.plotting.kspace_chern_plotting_functions as vp
import cPickle as pkl
import os
import time
import glob

'''functions for use with the KChernMagneticGyro class'''


def load_chern(kchern, verbose=True):
    """Load the chern results from disk

    Parameters
    ----------
    kchern : KChernMagneticGyro class instance
        kspace Chern calculation class

    Returns
    -------
    chern : KChernMagneticGyro class instance

    """
    savefn = get_cpfilename(kchern) + '.pkl'
    if verbose:
        print 'kchern_gyro_fns: Looking for filename: ' + savefn
    if os.path.isfile(savefn):
        with open(savefn, 'rb') as fn:
            chern = pkl.load(fn)

        if verbose:
            print 'kchern_gyro_fns: found chern on file, returning...'
        return chern
    else:
        if verbose:
            print 'kchern_gyro_fns: did not find chern on file, returning None...'
        return None


def calc_chern(kchern, verbose=True):
    """Compute the chern number using the wedge product of the projector

    Parameters
    ----------
    kchern : KChernMagneticGyro class instance
        kspace Chern calculation class

    Returns
    -------

    """
    # Unpack kchern a bit
    cp = kchern.cp
    # this is really the only function you need to change to do a different kind of lattice.
    # mM, vertex_points, od, fn, ar = lf.honeycomb_sheared(tvals, delta, phi, ons, base_dir)
    matk = mglatkfns.lambda_matrix_kspace(kchern.mgyro_lattice, eps=1e-10)
    bzvtcs = kchern.mgyro_lattice.lattice.get_bz(attribute=True)

    # set the directory for saving images and data.  The function above actually creates these directories for you.
    #  I know this isn't exaclty ideal.
    bzarea = dh.polygon_area(bzvtcs)
    # print 'bzvtcs = ', bzvtcs
    # print 'bzarea = ', bzarea
    # import matplotlib.pyplot as plt
    # plt.plot(bzvtcs[:, 0], bzvtcs[:, 1], 'b.-')
    # plt.show()
    # sys.exit()

    # Check if a similar results file exists --> new data is appended to the old data, if it exists
    savedir = cp['cpmeshfn']
    dio.ensure_dir(savedir)
    savefn = savedir + 'kspacechern_mgyro_density{0:07d}.pkl'.format(cp['density'])
    if verbose:
        print 'kchernmgfns: setting savefn = ', savefn
    globs = sorted(glob.glob(savedir + 'kspacechern_mgyro_density*.pkl'))
    # Obtain the filename with the density that is smaller than the requested one
    if globs:
        densities = np.array([int(globfn.split('density')[-1].split('.pkl')[0]) for globfn in globs])
        if (densities < cp['density']).any():
            biggestsmall = densities[densities < cp['density']][-1]
            smallerfn = savedir + 'kspacechern_mgyro_density{0:07d}.pkl'.format(biggestsmall)
            if verbose:
                print 'kchernmgfns: smallerfn = ', smallerfn
        else:
            if verbose:
                print 'kchernmgfns: no smaller density files are saved...'
            smallerfn = False
    else:
        smallerfn = False

    if verbose:
        print 'kchern_mgyro_fns: looking for saved file ' + savefn
    if os.path.isfile(savefn):
        if verbose:
            print 'kchern_mgyro_fns: chern file exists, overwriting: ', savefn
        # initialize empty lists to be filled up
        kkx, kky = [], []
        bands, traces = [], []
        npts = int(cp['density'] * bzarea)
    elif smallerfn:
        if verbose:
            print 'kchern_mgyro_fns: chern file with smaller density exists, loading to append...'
        with open(smallerfn, 'rb') as of:
            data = pkl.load(of)
            # Convert contents to lists so that we can append to them
            bands = list(data['bands'])
            kkx = list(data['kx'])
            kky = list(data['ky'])
            traces = list(data['traces'])

        # figure out how many points to append to reach desired density
        npts = int(cp['density'] * bzarea) - len(kkx)
    else:
        if verbose:
            print 'kchern_mgyro_fns: no chern file with smaller density exists... '
        # initialize empty lists to be filled up
        kkx, kky = [], []
        bands, traces = [], []
        npts = int(cp['density'] * bzarea)

    # Make the kx, ky points to add to current results
    kxy = dh.generate_random_xy_in_polygon(npts, bzvtcs, sorted=True)

    start_time = time.time()
    for ii in range(len(kxy)):
        # time for evaluating point index ii
        tpi = []
        # Display how much time is left
        if ii % 4000 == 1999 and verbose:
            end_time = time.time()
            tpi.append(abs((start_time - end_time) / (ii + 1)))
            total_time = np.mean(tpi) * len(kxy)
            printstr = 'Estimated time remaining: ' + '%0.2f s' % (total_time - (end_time - start_time))
            printstr += ', ii = ' + str(ii + 1) + '/' + str(len(kxy))
            print printstr

        eigval, tr = calc_bands(matk, kxy[ii, 0], kxy[ii, 1])
        kkx.append(kxy[ii, 0])
        kky.append(kxy[ii, 1])
        traces.append(tr)
        bands.append(eigval)

    bands = np.array(bands)
    traces = np.array(traces)
    kkx = np.array(kkx)
    kky = np.array(kky)

    bv = []
    if verbose:
        print 'kchern_mgyro_fns: np.shape(bands) = ', np.shape(bands)
        print 'kchern_mgyro_fns: np.shape(traces) = ', np.shape(traces)
    for ii in range(len(bands[0])):
        chern = ((1j / (2. * np.pi)) * np.mean(traces[:, ii]) * bzarea)  # chern numbers for bands
        bv.append(chern)

    dat_dict = {'kx': kkx, 'ky': kky,
                'chern': bv, 'bands': bands, 'traces': traces,
                'bzvtcs': bzvtcs}
    kchern.chern = dat_dict

    return kchern.chern


def save_chern(kchern, chern=None, save_png=True, verbose=True):
    """Save the result of computing the chern number for the bands

    Parameters
    ----------
    kchern : KChernMagneticGyro class instance
        kspace Chern calculation class
    chern : dict
        chern caclulation result, if not already in kchern.chern

    Returns
    -------

    """
    # Save the pickle with the data
    savefn = get_cpfilename(kchern)
    print 'saving results to file: ' + savefn + '.pkl'
    if chern is None:
        chern = kchern.chern

    with open(savefn + '.pkl', "wb") as fn:
        pkl.dump(chern, fn, protocol=pkl.HIGHEST_PROTOCOL)

    if verbose:
        # Show output on command line
        print 'bands = ', kchern.chern['bands']
        print 'np.shape(bands) = ', np.shape(kchern.chern['bands'])
        print 'chern = ', kchern.chern['chern']
        print 'np.shape(chern) = ', np.shape(kchern.chern['chern'])

    if save_png:
        print 'saving results to image'
        eps = 1e-10
        lat = kchern.mgyro_lattice.lattice
        tvals = np.unique(kchern.mgyro_lattice.OmK.ravel()[np.where(np.abs(lat.KL.ravel()) > eps)[0]])
        print 'tvals = ', tvals
        ons = kchern.mgyro_lattice.Omg
        vertex_points = chern['bzvtcs']
        kkx = chern['kx']
        kky = chern['ky']
        bands = chern['bands']
        traces = chern['traces']
        ax_lat, cc, lc, fig = vp.save_plot(kkx, kky, bands, traces, tvals, ons, vertex_points,
                                           od=savefn + '.png')


def get_cmeshfn(lp, rootdir=None):
    """Prepare the path where the cherns for a particular lattice are stored. If rootdir is specified, use that for the
    base of the path. Otherwise, adopt the base of the path from the lattice_params dict (lp) of the lattice.

    Parameters
    ----------
    lp : dict
        lattice parameters dictionary
    rootdir : str or None
        The root of the directory to which to save the chern calculation data

    Returns
    -------
    cmeshfn : str
    """
    meshfn_split = lp['meshfn'].split('/')
    ind = np.where(np.array(meshfn_split) == 'networks')[0][0]
    cmeshfn = ''
    if rootdir is None:
        for strseg in meshfn_split[0:ind]:
            cmeshfn += strseg + '/'
    else:
        cmeshfn = dio.prepdir(rootdir)
    cmeshfn += 'kspace_chern_magnetic_gyro/'
    for strseg in meshfn_split[(ind + 1):]:
        cmeshfn += strseg + '/'

    # Form physics subdir for interaction_range, aoverl, Omk, Omg, V0_pin_gauss, V0_spring_gauss
    cmeshfn += 'aol' + sf.float2pstr(lp['aoverl'], ndigits=8) + '/'
    if 'intrange' in lp['meshfn_exten']:
        cmeshfn += 'interaction_range{0:04d}'.format(lp['interaction_range']) + '/'

    eps = 1e-7
    if 'ABDelta' in lp:
        if np.abs(lp['ABDelta']) > eps:
            cmeshfn += 'ABd' + '{0:0.4f}'.format(lp['ABDelta']).replace('.', 'p').replace('-', 'n') + '/'
    if 'OmKspec' in lp:
        if lp['OmKspec'] != '':
            cmeshfn += lp['OmKspec']
        else:
            cmeshfn += 'Omk' + sf.float2pstr(lp['Omk'], ndigits=3)
    else:
        cmeshfn += 'Omk' + sf.float2pstr(lp['Omk'], ndigits=3)
    cmeshfn += '_Omg' + sf.float2pstr(lp['Omg'], ndigits=3)

    if 'pinconf' in lp:
        if lp['pinconf'] > 0:
            cmeshfn += '/pinconf' + '{0:04d}'.format(lp['pinconf'])

    ####################################
    # Now start final part of path's name
    print 'lp[V0_pin_gauss] = ', lp['V0_pin_gauss']
    if lp['V0_pin_gauss'] > 0 or lp['V0_spring_gauss'] > 0:
        dcdisorder = True
        cmeshfn += '/pinV' + sf.float2pstr(lp['V0_pin_gauss'], ndigits=4)
        cmeshfn += '_sprV' + sf.float2pstr(lp['V0_spring_gauss'], ndigits=4)
    else:
        dcdisorder = False

    return cmeshfn


def get_cpmeshfn(cp, lp):
    """Get the path for the specific chern calculation that uses the chern parameter dict cp on a mglat with lattice
    parameters lp.

    Parameters
    ----------
    cp : dict
        chern calculation parameters
    lp : dict
        lattice parameters
    """
    if 'rootdir' in cp:
        cpmeshfn = get_cmeshfn(lp, rootdir=cp['rootdir'])
        print '\n kfns: get_cpmeshfn(): rootdir is found in cp:'
        print 'cpmeshfn ==> ', cpmeshfn, '\n'
    else:
        print '\n kfns: get_cpmeshfn(): rootdir is NOT found in cp!'
        cpmeshfn = get_cmeshfn(lp)

    # Form cp subdir
    if cpmeshfn[-1] != '/':
        cpmeshfn += '_'

    cpmeshfn += cp['basis'] + '/'
    return cpmeshfn


def get_cpfilename(kchern):
    """Return the extension-free filename for the chern calculation"""
    return kchern.cp['cpmeshfn'] + 'kspacechern_mgyro_density{0:07d}'.format(kchern.cp['density'])


def calc_bands(matk, kxval, kyval):
    """calculates the eigenvalues and traces for some point kx, ky in the Bz

    Parameters
    ------------
    matk : lambda function

    kxval : float

    kyval : float


    Returns
    -------
    eigval : float array length 2*num_sites
         floats for band structure at input values of Kx and Ky
    tr : float array length 2*num_sites
         Berry curvature for bands in the BZ zone at kx, ky
    """
    h = 10 ** -5
    # func is a lambda function defined elsewhere in this module. (what a great name)
    der_func = lambda kvec: projeigvalvect(matk, kvec)
    # func (and therefore der_func) is a lambda function.
    p_v, eigval, eigvect = der_func([kxval, kyval])
    # print 'kchern_mgyro_fns.py: eigval = ', eigval
    # I think I just did this so that I would have one function of kx and ky to take the derivative of
    # in the calc_numerical.

    # calc numerical derivative in kx and ky
    fpx, fpy = amath.numerical_derivative_2d(der_func, h, [kxval, kyval])
    # print 'kchern_mgyro_fns.py: fpx = ', fpx
    # note: you could probably use a canned function to calculate the numerical derivative, but I didn't.

    tr = []
    for ii in range(len(p_v)):
        t1 = np.dot(fpx[ii], np.dot(p_v[ii], fpy[ii]))
        t2 = np.dot(fpy[ii], np.dot(p_v[ii], fpx[ii]))
        tr.append(np.trace(t1 - t2))

    # tr is what you integrate over the BZ to get the chern number.
    return eigval, tr


def projeigvalvect(matk, kvec):
    """compute projector, eigval, and eigvect"""
    eigval, eigvect = eig_vals_vects(matk(kvec), sort='imag', not_hermitian=False)
    p_v = calc_proj(eigvect)
    obj = [p_v, np.imag(eigval), eigvect]
    return obj


def calc_proj(eigvect):
    """Compute the projection operator for kspace chern calculation

    Parameters
    ----------
    eigvect

    Returns
    -------
    proj
    """
    proj = []
    lene = len(eigvect)
    for j in range(lene):
        proj.append(np.conjugate(np.reshape(eigvect[j], [lene, 1])) * np.reshape(eigvect[j], [1, lene]))

    return np.array(proj)
