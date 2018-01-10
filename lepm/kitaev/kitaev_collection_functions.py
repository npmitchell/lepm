import numpy as np
import lepm.lattice_elasticity as le
import lepm.plotting.plotting as leplt
import lepm.plotting.kitaev_plotting_functions as kpfns
import matplotlib.pyplot as plt
try:
    import cPickle as pickle
    import pickle as pl
except ImportError:
    import pickle
    import pickle as pl

"""
Functions supporting the KitaevCollection class
"""


def gap_midpoints_honeycomb(delta):
    """Get midpoint in energy of the gap for a honeycomb lattice with Omega_g = Omega_k

    Parameters
    ----------
    delta : lattice deformation angle in radians

    Returns
    -------
    midfreq : float
        The frequency in the center of the gap, computed from linear interpolation of band edge data
    """
    from scipy import interpolate
    print 'kcfns: interpolating: delta = ', delta
    mins = np.array([[1.200000000000000000e+02, 1.999774418229452122e+00],
                    [1.250000000000000000e+02, 2.000282666353143668e+00],
                    [1.300000000000000000e+02, 2.001554311155767429e+00],
                    [1.350000000000000000e+02, 2.003915844279629876e+00],
                    [1.400000000000000000e+02, 2.007416311659185659e+00],
                    [1.450000000000000000e+02, 2.012603123613059708e+00],
                    [1.500000000000000000e+02, 2.019149626543939924e+00],
                    [1.550000000000000000e+02, 2.028641334796071227e+00],
                    [1.600000000000000000e+02, 2.039851478127542084e+00],
                    [1.650000000000000000e+02, 2.054516514090857893e+00],
                    [1.700000000000000000e+02, 2.070945460207864297e+00],
                    [1.750000000000000000e+02, 2.093124186766038175e+00],
                    [1.800000000000000000e+02, 2.109658791108013798e+00],
                    [1.850000000000000000e+02, 2.094722397948975967e+00],
                    [1.900000000000000000e+02, 2.072124259261625578e+00],
                    [1.950000000000000000e+02, 2.054700225753695264e+00],
                    [2.000000000000000000e+02, 2.038659503829169939e+00],
                    [2.050000000000000000e+02, 2.027838517931622064e+00],
                    [2.100000000000000000e+02, 2.017076430256127928e+00],
                    [2.150000000000000000e+02, 2.010784767531016648e+00],
                    [2.200000000000000000e+02, 2.006081028042267622e+00],
                    [2.250000000000000000e+02, 2.003452762207305504e+00],
                    [2.300000000000000000e+02, 2.001560538610729356e+00],
                    [2.350000000000000000e+02, 2.000174272734998482e+00]])
    maxes = np.array([[1.200000000000000000e+02, 2.500256615038253916e+00],
                     [1.250000000000000000e+02, 2.497007956972428389e+00],
                     [1.300000000000000000e+02, 2.486196042678443519e+00],
                     [1.350000000000000000e+02, 2.467666057410397240e+00],
                     [1.400000000000000000e+02, 2.440388523864080561e+00],
                     [1.450000000000000000e+02, 2.406284060891525023e+00],
                     [1.500000000000000000e+02, 2.365221123134191750e+00],
                     [1.550000000000000000e+02, 2.320295945632198986e+00],
                     [1.600000000000000000e+02, 2.274521519035543449e+00],
                     [1.650000000000000000e+02, 2.229903095419124792e+00],
                     [1.700000000000000000e+02, 2.189786569154605012e+00],
                     [1.750000000000000000e+02, 2.153758910606546895e+00],
                     [1.800000000000000000e+02, 2.125455921485951638e+00],
                     [1.850000000000000000e+02, 2.152785860012932595e+00],
                     [1.900000000000000000e+02, 2.189105793927797805e+00],
                     [1.950000000000000000e+02, 2.231747790206741744e+00],
                     [2.000000000000000000e+02, 2.274523275299024760e+00],
                     [2.050000000000000000e+02, 2.320231679639985334e+00],
                     [2.100000000000000000e+02, 2.365189409373505303e+00],
                     [2.150000000000000000e+02, 2.405764446881530905e+00],
                     [2.200000000000000000e+02, 2.440490911937903462e+00],
                     [2.250000000000000000e+02, 2.467662836800375903e+00],
                     [2.300000000000000000e+02, 2.485946135285273861e+00],
                     [2.350000000000000000e+02, 2.497074387830856335e+00]])
    if np.abs(delta - 2.09439510239) < 1e-5:
        return np.array([2.25])
    else:
        min_interp = interpolate.interp1d(mins[:, 0], mins[:, 1])
        max_interp = interpolate.interp1d(maxes[:, 0], maxes[:, 1])
        midfreq = 0.5 * (min_interp(delta * 180./np.pi) + max_interp(delta * 180./np.pi))
        return np.array([midfreq])


def retrieve_param_value(value):
    if isinstance(value, float):
        return value
    elif isinstance(value, str):
        if 'gridlines' in value:
            # assume that the weak bond strength is varying
            return float(value.split('weak')[1].replace('n', '-').replace('p', '.'))
        else:
            raise RuntimeError('kcollfns.retrieve_param_value does not recognize this string type for a parameter')
    else:
        raise RuntimeError('kcollfns.retrieve_param_value does not recognize this param instance type for a parameter')


def nu_gradient_excitation(kcoll, glat_name, outdir=None, check=False):
    """Compute nu gradient sum in different directions weighted by the excitation amplitude of eigenvectors (gnu_psi2).

    Parameters
    ----------
    kcoll : KitaevCollection instance
    glat_name : str
        string specifier for the name of the GyroLattice. This is the key to the Chern.chern dictionary
    outdir : str or None
        path where to store the output pickle, if not None
    check : bool
        display intermediate results

    Returns
    -------
    gnu_psi2 : len(kszf) x len(eigval)/2 float array
        For each kitaev region size (row) and unique eigenvector (column), this is the sum of gradients in different
        directions of the spatially-resolved chern number, weighted by the displacement of gyroscopes in that
        eigenvector. The order of the evects is the same as the order for eigval[0:len(eigval)/2].
        Note: len(ksf) is len(kcoll.cherns[glat_name][0].chern_finsize[:, 1])
    xyvec :
    ksizes :
        chern_finsize[:, 3] for each chern, stacked: this is the characteristic width or size (diameter, width) of the
        kitaev summation region, in true units --> same units as lengths are measured in the lattice -- which is
        usually in median bond lengths.
    kszf :
        chern_finsize[:, 1] for each chern, stacked: this is the fractional number of particles in the sum (as a
        fraction of the # particles in the system) TIMES TWO!
    nugrids :
    nugsum :
    eigval :
    """
    from scipy import ndimage
    glat = kcoll.cherns[glat_name][0].gyro_lattice

    # Assume all cherns have the same len(ksize), so preallocate array
    # Make xynu, for which xynu[:,i] is the map of chern vals for the ith ksize
    # also make xyvec, so that xyvec[i] = [x, y] for computation xynu[i,:]
    kszf = kcoll.cherns[glat_name][0].chern_finsize[:, 1]
    xynu = np.zeros((len(kcoll.cherns[glat_name]), len(kszf)), dtype=float)
    ksizes = np.zeros((len(kcoll.cherns[glat_name]), len(kszf)), dtype=float)
    xyvec = np.zeros((np.shape(xynu)[0], 2), dtype=float)
    ind = 0
    for chernii in kcoll.cherns[glat_name]:
        # Add this chern to array of stored vals
        xx = float(chernii.cp['poly_offset'].split('/')[0])
        yy = float(chernii.cp['poly_offset'].split('/')[1])
        xynu[ind] = np.real(chernii.chern_finsize[:, -1])
        ksizes[ind] = chernii.chern_finsize[:, 2]
        xyvec[ind, :] = np.array([xx, yy])
        ind += 1

    if check:
        print 'xynu = ', xynu
        plt.imshow(xynu)
        plt.show()

    # Make grid of nu values, in fact one for each ksize
    # nugrids[i] is the grid of nu values for the ith ksize
    lenx = int(np.sqrt(len(kcoll.cherns[glat_name])))
    # print 'lenx = ', lenx
    # print 'len(kcoll.cherns[glat_name]) = ', len(kcoll.cherns[glat_name])
    nugrids = xynu.reshape((lenx, lenx, len(kszf))).T

    if check:
        print 'np.shape(xynu) = ', np.shape(xynu)
        tmp = nugrids[15, :, :]
        mappable = plt.imshow(tmp, interpolation='nearest', cmap='rwb0', vmin=-1, vmax=1)
        plt.colorbar()
        plt.title('chern values in space')
        plt.show()

    # nugsum is the sum of the individual gradients of nu in different directions.
    nugsum = np.zeros_like(nugrids, dtype=float)
    ind = 0
    for nugrid in nugrids:
        # For each ksize, get 2d gradient magnitude (summed in each direction separately
        diff = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]])
        nugsum[ind, :, :] = 0.25 * np.abs(ndimage.convolve(nugrid, diff, mode='reflect'))
        diff = np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]])
        nugsum[ind, :, :] += 0.25 * np.abs(ndimage.convolve(nugrid, diff, mode='reflect'))
        diff = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]])
        nugsum[ind, :, :] += 0.25 * np.abs(ndimage.convolve(nugrid, diff, mode='reflect'))
        diff = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]])
        nugsum[ind, :, :] += 0.25 * np.abs(ndimage.convolve(nugrid, diff, mode='reflect'))

        if check:
            print 'np.shape(nugrid) = ', np.shape(nugrid)
            leplt.plot_pcolormesh(xyvec[:, 0], xyvec[:, 1], nugrid.T.ravel(), 100,
                                  method='nearest', cmap='rwb0',
                                  vmin=-1.0, vmax=1.0, xlabel='x', ylabel='y',  cax_label=r'$\nu$',
                                  fontsize=12)
            le.movie_plot_2D(glat.lattice.xy, glat.lattice.BL, 0 * glat.lattice.BL[:, 0],
                             None, None, ax=plt.gca(), axcb=None, bondcolor='k',
                             colorz=False, ptcolor=None, figsize='auto',
                             colormap='BlueBlackRed', bgcolor='#ffffff', axis_off=False, axis_equal=True,
                             lw=0.2)
            kpfns.add_kitaev_regions_to_plot(kcoll.cherns[glat_name][0], ind, offsetxy=np.array([0, 0]))
            plt.pause(0.1)
            plt.clf()
            leplt.plot_pcolormesh(xyvec[:, 0], xyvec[:, 1], nugsum[ind].T.ravel(), 100,
                                  method='nearest', cmap='viridis',
                                  vmin=0.0, vmax=None, xlabel='x', ylabel='y',  cax_label=r'$|\nabla\nu|$',
                                  fontsize=12)
            le.movie_plot_2D(glat.lattice.xy, glat.lattice.BL, 0 * glat.lattice.BL[:, 0],
                             None, None, ax=plt.gca(), axcb=None, bondcolor='k',
                             colorz=False, ptcolor=None, figsize='auto',
                             colormap='BlueBlackRed', bgcolor='#ffffff', axis_off=False, axis_equal=True,
                             lw=0.2)
            kpfns.add_kitaev_regions_to_plot(kcoll.cherns[glat_name][0], ind, offsetxy=np.array([0, 0]))
            plt.pause(0.1)
            plt.clf()
        ind += 1

    # convert back to same order as xyvec for later
    # gsnu_vecs[:,i] is the vector of chern gradient magnitudes for the ith ksize, assoc with xyvec
    gsnu_vecs = nugsum.T.reshape(-1, len(kszf))

    # Smooth gradient -- I don't use this anymore
    # snugsum = np.zeros_like(nugsum, dtype=float)
    # for ind in range(len(nugsum)):
    #     snugsum[ind] = ndimage.uniform_filter(nugsum[ind], (2, 2))
    # sgsnu_vecs = snugsum.T.reshape(-1, len(kszf))

    if check:
        plt.imshow(nugrids[15, :, :], interpolation='none', cmap='rwb0', vmin=-1, vmax=1)
        plt.title(r'$\nu$')
        plt.show()

    # Associate each particle with an xy region
    # pt2loc's ith element is the index of xynu or nugrad assoc with ith gyro
    pt2loc = np.zeros_like(glat.lattice.xy[:, 0], dtype=int)
    ind = 0
    for xy in glat.lattice.xy:
        dist = np.abs(xyvec - xy)[:, 0]**2 + np.abs(xyvec - xy)[:, 1]**2
        loc = np.argmin(dist)
        pt2loc[ind] = loc
        ind += 1

    # Look at evects for this network. Compute L for each evect. Plot L as a function of evals.
    eigval, eigvect = glat.load_eigval_eigvect(attribute=True)
    gnu_psi2 = np.zeros((len(kszf), len(eigval)/2), dtype=float)
    for ind in range(int(len(eigval)*0.5)):
        if ind % 100 == 0:
            print 'Calculating locality for eigval ', ind, ' of ', len(eigval), '...'
        # Eigvect is stored as NModes x NP*2 array, with x and y components alternating, like:
        # x0, y0, x1, y1, ... xNP, yNP.
        mag1 = eigvect[ind]
        mag1x = np.array([mag1[2*i] for i in range(len(mag1)/2)])
        mag1y = np.array([mag1[2*i+1] for i in range(len(mag1)/2)])

        # Get magnitude of displacements
        mag2 = np.array([abs(mag1x[i])**2 + abs(mag1y[i])**2 for i in range(len(mag1x))]).flatten()
        # gnu_psi2 is gradsum of nu weighted by magnitude of displacements of eigenvector
        gnu_psi2[:, ind] = np.array([np.sum(mag2 * np.abs(gsnu_vecs[pt2loc, kk])) / np.sum(mag2)
                                    for kk in range(len(kszf))])

    if outdir is not None:
        # Save results
        lld = {'gnu_psi2': gnu_psi2, 'ksizes': ksizes, 'kszf': kszf, 'xyvec': xyvec, 'nugrids': nugrids,
               'eigval': eigval, 'nugsum': nugsum}
        outfn = outdir + '_gnu_dict.pkl'
        with open(outfn, 'wb') as fn:
            pickle.dump(lld, fn)

    return gnu_psi2, xyvec, ksizes, kszf, nugrids, nugsum, eigval
