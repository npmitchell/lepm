import numpy as np
import lepm.dataio as dio
import matplotlib.pyplot as plt
import lepm.plotting.plotting as leplt
import lepm.stringformat as sf
import lepm.math_functions as amath
from lepm.lattice_elasticity import eig_vals_vects
import lepm.haldane.haldane_lattice_kspace_functions as hlatkfns
import lepm.chern.plotting.kchern_haldane_plotting_fns as kchernhpfns
import lepm.data_handling as dh
# import lepm.plotting.kspace_chern_plotting_functions as vp
import cPickle as pkl
import os
import time
import glob
import sys
import lepm.gyro_data_handling as gdh
import lepm.chern.plotting.kchern_gyro_plotting_fns as kcherngpfns

'''Module of standalone functions for computing chern numbers with generalized matrices'''


def prepare_generalized_dispersion_params(lat, kx=None, ky=None, nkxvals=50, nkyvals=20, outdir=None, name=None):
    """Prepare the filename and kx, ky for computing dispersion relation

    Returns
    """
    if not lat.lp['periodicBC']:
        raise RuntimeError('Cannot compute dispersion for open BC system')
    elif lat.lp['periodic_strip']:
        print 'Evaluating infinite-system dispersion for strip: setting ky=constant=0.'
        ky = [0.]
        bboxx = max(lat.lp['BBox'][:, 0]) - min(lat.lp['BBox'][:, 0])
        minx, maxx = -1. / bboxx, 1. / bboxx
    elif ky is None or kx is None:
        bzvtcs = lat.get_bz(attribute=True)
        minx, maxx = np.min(bzvtcs[:, 0]), np.max(bzvtcs[:, 0])
        miny, maxy = np.min(bzvtcs[:, 1]), np.max(bzvtcs[:, 1])

    if kx is None:
        kx = np.linspace(minx, maxx, nkxvals, endpoint=True)

    if ky is None:
        ky = np.linspace(miny, maxy, nkyvals, endpoint=True)

    # First check for saved dispersion
    if outdir is None:
        try:
            outdir = dio.prepdir(lat.lp['meshfn'])
        except:
            outdir = './'
    else:
        outdir = dio.prepdir(outdir)

    if name is None:
        try:
            name = 'generalized_dispersion' + lat.lp['meshfn_exten'] + '_nx' + str(len(kx)) + '_ny' + str(len(ky))
        except:
            name = 'generalized_dispersion' + '_nx' + str(len(kx)) + '_ny' + str(len(ky))
        name += '_maxkx{0:0.3f}'.format(np.max(np.abs(kx))).replace('.', 'p')
        name += '_maxky{0:0.3f}'.format(np.max(np.abs(ky))).replace('.', 'p')

    name = outdir + name
    return name, kx, ky


def infinite_dispersion_matk(matk, lat, kx=None, ky=None, nkxvals=50, nkyvals=20, save=True, save_plot=True,
                             title='matrix dispersion relation', outdir=None, name=None, ax=None, lwscale=1., verbose=False):
    """

    Parameters
    ----------
    glat
    kx
    ky
    nkxvals
    nkyvals
    save
    save_plot
    title
    outdir
    name
    ax
    lwscale
    verbose

    Returns
    -------

    """
    name, kx, ky = prepare_generalized_dispersion_params(lat, kx=kx, ky=ky, nkxvals=nkxvals, nkyvals=nkyvals,
                                                         outdir=outdir, name=name)
    print('checking for file: ' + name + '.pkl')
    if glob.glob(name + '.pkl'):
        saved = True
        with open(name + '.pkl', "rb") as fn:
            res = pkl.load(fn)

        omegas = res['omegas']
        kx = res['kx']
        ky = res['ky']
    else:
        # dispersion is not saved, compute it!
        saved = False
        omegas = np.zeros((len(kx), len(ky), len(matk([0, 0]))))
        ii = 0
        for kxi in kx:
            if ii % 50 == 0:
                print 'glatkspace_fns: infinite_dispersion(): ii = ', ii
            jj = 0
            for kyj in ky:
                # print 'jj = ', jj
                matrix = matk([kxi, kyj])
                # print 'glatkspace_fns: diagonalizing...'
                eigval, eigvect = np.linalg.eig(matrix)
                si = np.argsort(np.imag(eigval))
                omegas[ii, jj, :] = np.imag(eigval[si])
                jj += 1
            ii += 1

    if save_plot or ax is not None:
        if ax is None:
            fig, ax = leplt.initialize_1panel_centered_fig(Hfig=90, wsfrac=0.6)

        for jj in range(len(ky)):
            for kk in range(len(omegas[0, jj, :])):
                ax.plot(kx, omegas[:, jj, kk], 'k-', lw=lwscale * max(0.03, 5. / (len(kx) * len(ky))))
        ax.set_title(title)
        ax.set_xlabel(r'$k$ $[\langle \ell \rangle ^{-1}]$')
        ax.set_ylabel(r'$\omega$')
        ylims = ax.get_ylim()
        ax.set_ylim(0, ylims[1])
        # Save the plot
        if save_plot:
            print 'saving ' + name + '.png'
            plt.savefig(name + '.png', dpi=200)
            plt.close('all')

    if save:
        if not saved:
            res = {'omegas': omegas, 'kx': kx, 'ky': ky}
            with open(name + '.pkl', "wb") as fn:
                pkl.dump(res, fn)

    return omegas, kx, ky


def prepare_chern_filename(cp, appendstr=None):
    """Create the filename (path) to load/save chern calculation.
    Update cp with essential parameters (deriv_res) if missing.
    Also create filename for searching for identical chern calulation with lower density.

    Parameters
    ----------

    Returns
    -------
    savefn : str
    cp : dict
    search_density_fn : str
    """
    savedir = cp['cpmeshfn']
    savefn = savedir + 'kspacechern_density{0:07d}'.format(cp['density'])
    search = savedir + 'kspacechern_density*'
    if cp['ortho']:
        savefn += '_ortho'
        search += '_ortho'
    if 'deriv_res' in cp:
        if cp['deriv_res'] != 1e-5:
            savefn += '_dres{0:0.3e}'.format(cp['deriv_res']).replace('-', 'n').replace('.', 'p')
            search += '_dres{0:0.3e}'.format(cp['deriv_res']).replace('-', 'n').replace('.', 'p')
    else:
        cp['deriv_res'] = 1e-5

    if appendstr is not None:
        savefn += appendstr
        search += appendstr

    savefn += '.pkl'
    search += '.pkl'
    cp['savefn'] = savefn
    search_density_fn = search
    return savefn, cp, search_density_fn


def load_chern(cp, appendstr=None, verbose=True):
    """Load the chern results from disk

    Parameters
    ----------
    kchern : KChernGyro class instance
        kspace Chern calculation class

    Returns
    -------

    """
    savefn, cp, search_density_fn = prepare_chern_filename(cp, appendstr=appendstr)

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


def get_proj(matk, cp, kvec=np.array([0, 0]), signed_norm=False):
    """

    Parameters
    ----------
    kchern : instance of KChernGyro class
    kvec : 2 x 1 float array or length=2 list of floats

    Returns
    -------
    proj :
        projection operator
    eigval :
        eigenvalues of the kspace matrix at the supplied kvec
    eigvect :
        eigenvectors of the kspace matrix at the supplied kvec
    matk :
        lambda function of the dynamical matrix for input 'kvec'

    """
    # matk = glatkfns.lambda_matrix_kspace(kchern.gyro_lattice, eps=1e-10)
    # print 'kchern_gyro_functions: matk(kvec) = ', matk(kvec)
    eigval, eigvect = eig_vals_vects(matk(kvec), sort='imag', not_hermitian=False)
    proj = calc_proj(eigvect, eigval, ortho=cp['ortho'], signed_norm=signed_norm)
    # print 'proj = ', proj
    # import lepm.plotting.plotting as leplt
    # leplt.plot_complex_matrix(proj, outpath='/Users/npmitchell/Desktop/proj_.png')
    # sys.exit()
    if cp['ortho']:
        eigvect = gdh.orthonormal_eigvect(eigvect, negative=signed_norm)

    return proj, matk(kvec), eigval, eigvect


def calc_berry(matk, cp, kxy):
    """Compute the berry curvature at each supplied wavevector

    Parameters
    ----------
    kchern : KChernGyro instance
    kxy : n x 2 float array
        wavevectors at which to evaluate the berry curvature

    Returns
    -------
    berry_dict : dict
    """
    deriv_res = cp['deriv_res']
    kkx, kky = [], []
    bands, traces = [], []
    # matk = glatkfns.lambda_matrix_kspace(kchern.gyro_lattice, eps=1e-10)
    for ii in range(len(kxy)):
        eigval, tr = calc_bands(matk, kxy[ii, 0], kxy[ii, 1], h=deriv_res)
        kkx.append(kxy[ii, 0])
        kky.append(kxy[ii, 1])
        traces.append(tr)
        bands.append(eigval)

    bands = np.array(bands)
    traces = np.array(traces)
    kkx = np.array(kkx)
    kky = np.array(kky)
    berry_dict = {'kx': kkx, 'ky': kky, 'bands': bands, 'traces': traces}
    return berry_dict


def calc_chern(matk, cp, lattice, appendstr=None,
               verbose=True, bz_cutoff=1e14, kxy=None, overwrite=False, signed_norm=False):
    """Compute the chern number using the wedge product of the projector

    Parameters
    ----------
    kchern : KChernGyro class instance
        kspace Chern calculation class
    bz_cutoff : float
        if BZ has a dimension larger than bz_cutoff, then we return zero chern number and no bands

    Returns
    -------
    kchern.chern
    """
    # this is really the only function you need to change to do a different kind of lattice.
    # mM, vertex_points, od, fn, ar = lf.honeycomb_sheared(tvals, delta, phi, ons, base_dir)
    # matk = glatkfns.lambda_matrix_kspace(kchern.gyro_lattice, eps=1e-10)
    bzvtcs = lattice.get_bz(attribute=True)

    # Define the brillouin zone polygon
    bzarea = dh.polygon_area(bzvtcs)

    # Check if a similar results file exists --> new data is appended to the old data, if it exists
    savedir = cp['cpmeshfn']
    dio.ensure_dir(savedir)
    savefn, cp, search_density_fn = prepare_chern_filename(cp, appendstr=appendstr)
    print 'established savefn: ', cp['savefn']

    if kxy is None:
        globs = sorted(glob.glob(search_density_fn))
        # Obtain the filename with the density that is smaller than the requested one
        if globs and not overwrite:
            densities = np.array([int(globfn.split('density')[-1].split('.pkl')[0].split('_')[0]) for globfn in globs])
            if (densities < cp['density']).any():
                biggestsmall = densities[densities < cp['density']][-1]
                smallerfn = savedir + 'kspacechern_density{0:07d}'.format(biggestsmall)
                if cp['ortho']:
                    smallerfn += '_ortho'
                if 'deriv_res' in cp:
                    if cp['deriv_res'] != 1e-5:
                        smallerfn += '_dres{0:0.3e}'.format(cp['deriv_res']).replace('-', 'n').replace('.', 'p')

                smallerfn += appendstr
                smallerfn += '.pkl'
            else:
                smallerfn = False
        else:
            smallerfn = False

        if verbose:
            print 'kchern_gyro_fns: looking for saved file ' + savefn
        if os.path.isfile(savefn) and overwrite:
            if verbose:
                print 'kchern_gyro_fns: chern file exists, overwriting: ', savefn
            # initialize empty lists to be filled up
            kkx, kky = [], []
            bands, traces = [], []
            npts = int(cp['density'] * bzarea)
        elif os.path.isfile(savefn):
            # File of same size exists
            chern = load_chern(cp, appendstr=appendstr, verbose=True)
            print 'generalized_kchern_fns: loaded chern, returning chern'
            return chern
        elif smallerfn:
            if verbose:
                print 'kchern_gyro_fns: chern file with smaller density exists, loading to append...'
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
                print 'generalized_kchern_fns: no chern file with smaller density exists, computing from scratch...'
            # initialize empty lists to be filled up
            kkx, kky = [], []
            bands, traces = [], []
            npts = int(cp['density'] * bzarea)

        # Make the kx, ky points to add to current results
        # Handle cases where the BZ is too elongated to populate in reasonable time
        try:
            kxy = dh.generate_random_xy_in_polygon(npts, bzvtcs, sorted=True)
            if len(kxy) == 0:
                if (bzvtcs > bz_cutoff).any():
                    print 'The BZ is too large to fill with kvec points'
                    kkx, kky = np.zeros(10), np.zeros(10)
                    bv = np.zeros(2 * len(lattice.xy[:, 0]))
                    bands = np.nan * np.ones(2 * len(lattice.xy[:, 0]))
                    traces = np.nan * np.ones((2 * len(lattice.xy[:, 0]), 10))
                    dat_dict = {'kx': kkx, 'ky': kky,
                                'chern': bv, 'bands': bands, 'traces': traces,
                                'bzvtcs': bzvtcs}
                    chern = dat_dict
                    return chern
                else:
                    print 'generalized_kchern_fns: bzvtcs = ', bzvtcs
                    print 'generalized_kchern_fns: kxy = ', kxy
                    raise RuntimeError(
                        'Could not generate random xy in polygon, but BZ is not larger than cutoff in any dim.')
        except ValueError:
            print 'The BZ is too large to fill with kvec points'
            if (np.abs(bzvtcs) > bz_cutoff).any():
                print 'The BZ is too large to fill with kvec points'
                kkx, kky = np.zeros(10), np.zeros(10)
                bv = np.zeros(2 * len(lattice.xy[:, 0]))
                bands = np.nan * np.ones(2 * len(lattice.xy[:, 0]))
                traces = np.nan * np.ones((2 * len(lattice.xy[:, 0]), 10))
                dat_dict = {'kx': kkx, 'ky': kky,
                            'chern': bv, 'bands': bands, 'traces': traces,
                            'bzvtcs': bzvtcs}
                chern = dat_dict
                return chern
            else:
                print 'kchern_gyro_fns: bzvtcs = ', bzvtcs
                print 'bz_cutoff = ', bz_cutoff
                # print 'generalized_kchern_fns: kxy = ', kxy
                raise RuntimeError(
                    'Could not generate random xy in polygon, but BZ is not larger than cutoff in any dim.')
                dat_dict = {'kx': kkx, 'ky': kky,
                            'chern': bv, 'bands': bands, 'traces': traces,
                            'bzvtcs': bzvtcs}
    else:
        # kxy is supplied -- compute at those locations
        print 'kxy is supplied, computing berry at supplied points'
        kkx, kky = [], []
        bands, traces = [], []

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

        eigval, tr = calc_bands(matk, kxy[ii, 0], kxy[ii, 1], h=cp['deriv_res'], ortho=cp['ortho'],
                                signed_norm=signed_norm)
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
        print 'kchern_gyro_fns: np.shape(bands) = ', np.shape(bands)
        print 'kchern_gyro_fns: np.shape(traces) = ', np.shape(traces)
    for ii in range(len(bands[0])):
        chern = ((1j / (2. * np.pi)) * np.mean(traces[:, ii]) * bzarea)  # chern numbers for bands
        bv.append(chern)

    dat_dict = {'kx': kkx, 'ky': kky,
                'chern': bv, 'bands': bands, 'traces': traces,
                'bzvtcs': bzvtcs}
    chern = dat_dict

    return chern


def save_chern(chern_dict, savefn, save_png=True, cmap='bbr0'):
    """Save the result of computing the chern number for the bands

    Parameters
    ----------
    kchern : KChernGyro class instance
        kspace Chern calculation class
    chern : dict
        chern caclulation result, if not already in kchern.chern

    Returns
    -------

    """
    # Strip .pkl from the name
    if savefn[-4:] == '.pkl':
        savefn = savefn[0:-4]
    # Save the pickle with the data
    print 'saving results to file: ' + savefn + '.pkl'
    with open(savefn + '.pkl', "wb") as fn:
        pkl.dump(chern_dict, fn, protocol=pkl.HIGHEST_PROTOCOL)

    if save_png:
        imgfn = savefn + '.png'
        save_chern_png(chern_dict, cp, imgfn=imgfn, cmap=cmap)


def save_chern_png(chern_datdict, cp, imgfn=None, vmin=-1.0, vmax=1.0, cmap='bbr0'):
    """Save a png of the band structure colored by chern number. Default is for colorbar to range from -1 to 1.

    Parameters
    ----------
    chern_datdict : dict
        equivalent to KChernGyro.chern for generalized dynamical matrices
    cp : dict
        chern parameters dictionary
    imgfn : str or None
        the path where an image of the band structure is stored

    Returns
    -------
    fig, ax, cax : matplotlib Figure instance, axis instance, colorbar axis instance
    """
    print 'saving results to image'
    if imgfn is None:
        imgfn = cp['savefn'] + '.png'

    kchern = {'chern': chern_datdict}
    fig, ax, cax = kcherngpfns.plot_chernbands(kchern, fig=None, ax=None, cax=None, outpath=imgfn,
                                               eps=1e-10, cmap=cmap,
                                               vmin=vmin, vmax=vmax, round_chern=False)
    return fig, ax, cax


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
    cmeshfn += 'kspace_chern_gyro/'
    for strseg in meshfn_split[(ind + 1):]:
        cmeshfn += strseg + '/'

    # Form physics subdir for Omk, Omg, V0_pin_gauss, V0_spring_gauss
    eps = 1e-7

    if 'ABDelta' in lp:
        if np.abs(lp['ABDelta']) > eps:
            cmeshfn += 'ABd' + '{0:0.4f}'.format(lp['ABDelta']).replace('.', 'p').replace('-', 'n') + '/'
    if 'pinconf' in lp:
        if lp['pinconf'] > 0:
            cmeshfn += 'pinconf' + '{0:04d}'.format(lp['pinconf']) + '/'

    if 'OmKspec' in lp:
        if lp['OmKspec'] != '':
            cmeshfn += lp['OmKspec']
        else:
            cmeshfn += 'Omk' + sf.float2pstr(lp['Omk'])
    else:
        cmeshfn += 'Omk' + sf.float2pstr(lp['Omk'])

    cmeshfn += '_Omg' + sf.float2pstr(lp['Omg'])
    if lp['V0_pin_gauss'] > 0 or lp['V0_spring_gauss'] > 0:
        cmeshfn += '_pinV' + sf.float2pstr(lp['V0_pin_gauss'])
        if np.abs(lp['V0_spring_gauss']) < eps:
            if 'pinconf' in lp:
                if lp['pinconf'] > 0:
                    # Add pinconf tag first, then spring disorder strength
                    cmeshfn += '_pinconf' + '{0:04d}'.format(lp['pinconf'])

            cmeshfn += '_sprV' + sf.float2pstr(lp['V0_spring_gauss'])
        else:
            # Add spring disorder strength FIRST, then configuration tag
            cmeshfn += '_sprV' + sf.float2pstr(lp['V0_spring_gauss'])
            cmeshfn += '_pinconf' + '{0:04d}'.format(lp['pinconf'])

    if 'ABDelta' in lp:
        if lp['ABDelta'] > 0:
            cmeshfn += '_ABd' + sf.float2pstr(lp['ABDelta'])
    return cmeshfn


def get_cpmeshfn(cp, lp):
    """Get the path for the specific chern calculation that uses the chern parameter dict cp on a glat with lattice
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
    cpmeshfn += '_' + cp['basis']

    if cp['ortho']:
        cpmeshfn += '_ortho'

    cpmeshfn += '/'
    return cpmeshfn


def get_cpfilename(cp):
    """Return the extension-free filename for the chern calculation"""
    savefn = cp['cpmeshfn'] + 'kspacechern_density{0:07d}'.format(cp['density'])
    if cp['ortho']:
        savefn += '_ortho'
    if 'deriv_res' in cp:
        if cp['deriv_res'] != 1e-5:
            savefn += '_dres{0:0.3e}'.format(cp['deriv_res']).replace('-', 'n').replace('.', 'p')
    return savefn


def calc_bands(matk, kxval, kyval, h=1e-5, ortho=True, signed_norm=False):
    """calculates the eigenvalues and traces (berry curvature) for some point kx, ky in the Bz

    Parameters
    ------------
    matk : lambda function
        the function which provides the dynamical matrix with input of [kx, ky]
    kxval : float
        the value of kx at which to compute bands
    kyval : float
        the value of ky at which to compute bands
    h : float
        small value relative to the dimensions of the BZ

    Returns
    -------
    eigval : float array length 2*num_sites
         floats for band structure at input values of Kx and Ky
    tr : float array length 2*num_sites
         Berry curvature for bands in the BZ zone at kx, ky
    """
    der_func = lambda kvec: projeigvalvect(matk, kvec, ortho=ortho, signed_norm=signed_norm)
    # func (and therefore der_func) is a lambda function.
    p_v, eigval, eigvect = der_func([kxval, kyval])
    # print 'kchern_gyro_fns.py: eigval = ', eigval
    # I think I just did this so that I would have one function of kx and ky to take the derivative of
    # in the calc_numerical.

    # calc numerical derivative in kx and ky
    fpx, fpy = amath.numerical_derivative_2d(der_func, h, [kxval, kyval])
    # print 'kchern_gyro_fns.py: fpx = ', fpx
    # note: you could probably use a canned function to calculate the numerical derivative, but I didn't.

    tr = []
    for ii in range(len(p_v)):
        t1 = np.dot(fpx[ii], np.dot(p_v[ii], fpy[ii]))
        t2 = np.dot(fpy[ii], np.dot(p_v[ii], fpx[ii]))
        tr.append(np.trace(t1 - t2))

    # tr is what you integrate over the BZ to get the chern number.
    # print 'kchern_gyro_functions: exiting here'
    # print 'eigval = ', eigval
    # print 'tr = ', tr
    # sys.exit()
    return eigval, tr


def projeigvalvect(matk, kvec, ortho=True, signed_norm=False):
    """compute projector, eigval, and eigvect

    Parameters
    ----------
    matk
    kvec
    ortho

    Returns
    -------
    obj : list of projector, Im(eigvals), and eigvect
    """
    eigval, eigvect = eig_vals_vects(matk(kvec), sort='imag', not_hermitian=False)
    p_v = calc_proj(eigvect, eigval, ortho=ortho, signed_norm=signed_norm)
    obj = [p_v, np.imag(eigval), eigvect]
    return obj


def calc_proj(eigvect, eigval, ortho=True, eps=1e-7, verbose=False, signed_norm=False):
    """Compute the projection operator for kspace chern calculation

    Parameters
    ----------
    eigvect

    Returns
    -------
    proj : outer product
    """
    if ortho:
        eigvect = gdh.orthonormal_eigvect(eigvect, negative=signed_norm)

    if (np.real(eigval) / np.max(np.imag(eigval)) < eps).all():
        if verbose:
            print 'kchern_gyro_functions: considering only imaginary component of eigval'
        eigval = np.imag(eigval)
    elif (np.imag(eigval) / np.max(np.real(eigval)) < eps).all():
        if verbose:
            print 'kchern_gyro_functions: considering only real component of eigval'
        eigval = np.real(eigval)
    else:
        raise RuntimeError('Eigenvalues are neither real nor complex, but comparable mixture.')

    proj = []
    lene = len(eigvect)

    # Note:
    # if mm[:int(0.5 * lene)] *= -1 and pj = np.dot(pj, mm) * np.sign(eigval[j]), get same results as before
    # if mm[int(0.5 * lene):] *= -1 and pj = np.dot(pj, mm) * np.sign(-eigval[j]), get same results as before
    # Here, define Proj using M with on diagonals
    mm = np.identity(lene)
    # Old convention: pre 2018-09-27
    # mm[:int(0.5 * lene)] *= -1
    # New convention: post 2018-09-27
    mm[int(0.5 * lene):] *= -1
    # Here, define proj using Q with off diagonals
    # qq = np.zeros((lene, lene))

    # Check mm
    # import matplotlib.pyplot as plt
    # plt.clf()
    # hh = plt.imshow(mm)
    # plt.colorbar(hh)
    # print 'eigval = ', eigval
    # plt.show()
    # sys.exit()

    for j in range(lene):
        pj = np.conjugate(np.reshape(eigvect[j], [lene, 1])) * np.reshape(eigvect[j], [1, lene])
        if ortho:
            pj = np.dot(pj, mm) * np.sign(eigval[j])

        proj.append(pj)

    proj = np.array(proj)
    # print 'np.shape(proj) = ', np.shape(proj)
    # p2mp = np.dot(proj, proj) - proj
    # import matplotlib.pyplot as plt
    # plt.clf()
    # plt.imshow(np.abs(p2mp))
    # plt.show()
    # sys.exit()
    return proj


def plot_berryband(kchern, cbar_ticks=None, title=None, ax=None, vmin=None, vmax=None, alpha=1.0):
    """Plot the berry curvature of an indicated band on the axis, ax

    Parameters
    ----------
    kchern : KChernGyro instance

    Returns
    -------
    fig, ax, cax
    """
    if ax is None:
        fig, ax, cax = leplt.initialize_1panel_cbar_fig(Wfig=180, wsfrac=0.5, tspace=10, y0frac=0.08, x0frac=0.15)

    vertex_points = kchern.chern['bzvtcs']
    kkx = kchern.chern['kx']
    kky = kchern.chern['ky']
    bands = kchern.chern['bands']
    cherns = kchern.chern['chern']
    berry = kchern.chern['traces']
    # b = ((1j / (2 * np.pi)) * np.mean(traces[:, i]) * bzarea)

    # Get index and parameter value of this glat
    paramval = retrieve_param_value(kchern.gyro_lattice.lp[param])
    index = np.where(paramval == params)[0][0]

    if isinstance(cmap, str):
        cmap = lecmaps.ensure_cmap(cmap)

    # Create int array indexing bands to plot, but start halfway up and only do positive eigval bands
    todo = np.arange(int(0.5 * len(bands[0, :])), len(bands[0, :]))
    for kk in todo:
        berrykk = berry[:, kk]

        # Plot berry curvature as heatmap
        xx, yy, zz = dh.interpol_meshgrid(kkx, kky, np.imag(berrykk), ngrid, method='nearest')
        ax.pcolormesh(xx, yy, zz, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)

        # Also draw the BZ
        patch_handle = mask_ax_to_bz(ax, vertex_points)
        # poly = Polygon(vertex_points, closed=True, fill=True, lw=1, alpha=1.0, facecolor='none', edgecolor='k')
        # ax.add_artist(poly)

        # Add title
        ax.set_xlabel('wavenumber, $k_x$')
        ax.set_ylabel('wavenumber, $k_y$')
        if not titlelock:
            title = r'Berry curvature, $\nu = $' + \
                    '{0:0.1f}'.format(float(np.real(cherns[kk]))) + ' ' + \
                    paramlabel + '=' + '{0:0.2f}'.format(float(paramval))
        ax.text(0.5, 1.04, title, transform=ax.transAxes, ha='center', va='center')

        # Do colorbar
        if cbar_ticks is None:
            cbar_ticks = [vmin, vmax]

        sm = leplt.empty_scalar_mappable(cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(sm, cax=cax, ticks=cbar_ticks, label=r'Berry curvature, $\mathcal{F}$', alpha=alpha)

        # xy limits
        vpr = vertex_points.ravel()
        ax.set_xlim(np.min(vpr), np.max(vpr))
        ax.set_ylim(np.min(vpr), np.max(vpr))

    return fig, ax, cax
