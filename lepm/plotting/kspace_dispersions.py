import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
import lepm.data_handling as dh

'''
'''


def periodicstrip(kx, bands, ax=None, colorgaps=False, gaps=None, gapthres=4, de=0.1, eps_exx=0.01, lw0=0.1, lw1=1.0,
                  upcolor="#55a254", downcolor="#c042c8", ingap_thres=0.1):
    """

    Parameters
    ----------
    kx
    bands
    colorgaps : bool
    gaps : #gaps x 2 float array or None
    gapthres : float or int
        threshold number of curves in each bin to consider it a gap
    lw0 : float
        Line width for non-edge mode states
    lw1 : float
        Line width for edge mode states
    upcolor : color specifier
    downcolor : color specifier

    Returns
    -------

    """
    # Instantiate axis if needed
    if ax is None:
        ax = plt.gca()

    # Make sure we're using the right axis of 'bands'
    if np.shape(bands)[1] != len(kx):
        bands = bands.T

    if colorgaps:
        gaps = estimate_gaps_from_bands(bands, gapthres, de)

    ind = 0
    for band in bands:
        ax.plot(kx, band, 'k-', lw=lw0)

        # Color based on presence of gaps
        if colorgaps and (frac_band_in_range(band, gaps) > ingap_thres).any():
            # The above checks that a substantial part of the band is in the gap (more than ingap_thres)
            for gap in gaps:
                ingap = np.where(np.logical_and(band > gap[0], band < gap[1]))[0]

                if len(ingap) > 2:
                    # For each contiguous range of indices, find where slope is up or down and color accordingly
                    segments = dh.consecutive(ingap, stepsize=1)

                    for segment in segments:
                        print 'segment = ', segment
                        # Find where slope is positive
                        kseg = kx[segment]
                        bseg = band[segment]
                        up = np.where(np.diff(bseg) > 0)[0]
                        down = np.where(np.diff(bseg) < 0)[0]
                        # Include endpoints that are missed by diff, except if they run out of range
                        up = np.setdiff1d(np.unique(np.hstack((up + 1, up))), np.array([-1, len(segment)]))
                        down = np.setdiff1d(np.unique(np.hstack((down + 1, down))), np.array([-1, len(segment)]))

                        upsegs = dh.consecutive(up, stepsize=1)
                        downsegs = dh.consecutive(down, stepsize=1)

                        # For each up or down portion, get contiguous segment
                        if len(upsegs[0]) > 0:
                            print 'upsegs = ', upsegs
                            for upseg in upsegs:
                                # Here we set a minimum length of the curve that must exist in the gap,
                                # not just a couple pts
                                if len(upseg) > 2:
                                    ax.plot(kseg[upseg], bseg[upseg], color=upcolor, lw=lw1)
                        if len(downsegs[0]) > 0:
                            print 'downsegs = ', downsegs
                            for downseg in downsegs:
                                # Here we set a minimum length of the curve that must exist in the gap,
                                # not just a couple pts
                                if len(upseg) > 2:
                                    ax.plot(kseg[downseg], bseg[downseg], color=downcolor, lw=lw1)

            # User gradients to find line crossings
            # grad = np.gradient(np.gradient(band))
            # if (np.abs(grad) > eps_exx).any():
            #     # This is an edgemode since it intersects with another band (Normal eigvals repel, but here there is
            #     # a singularity at the crossing)
            #     # Plot the positive part as upcolor, other part as downcolor
            #     dips = np.where(grad < - eps_exx)[0]
            #     peaks = np.where(grad > eps_exx)[0]
            #     # make a list of all the singularities
            #     sings = np.hstack((dips, peaks))
            #     print 'dips = ', dips
            #     print 'peaks = ', peaks
            #
            #     start = 0
            #     for sing in sings:
            #         # is the first singularity a dip or a peak
            #         if sing in dips:
            #             ax.plot(kx[start:sing + 1], band[start:sing + 1], color=downcolor, lw=lw1)
            #         elif sing in peaks:
            #             ax.plot(kx[start:sing + 1], band[start:sing + 1], color=upcolor, lw=lw1)
            #         start = sing
            #
            #     if start in dips:
            #         ax.plot(kx[start:], band[start:], color=upcolor, lw=lw1)
            #     else:
            #         ax.plot(kx[start:], band[start:], color=downcolor, lw=lw1)

    return ax


def band_in_range(band, gaps):
    """Check if a band lives in an energy range, such as a gap

    Parameters
    ----------
    band : N x 1 float array
        A list of energies for different k's
    gaps : #ranges x 2 float array
        The ranges of energy values to look in

    Returns
    -------
    ingap : bool
        Whether the band is in each gap
    """
    ingap = []
    for gap in gaps:
        if np.logical_and((band > gap[0]).any(), (band < gap[1]).any()):
            ingap.append(True)
        else:
            ingap.append(False)

    return np.array(ingap)


def frac_band_in_range(band, gaps):
    """Measure how much of a band is in each gap

    Parameters
    ----------
    band : N x 1 float array
        A list of energies for different k's
    gaps : #ranges x 2 float array
        The ranges of energy values to look in

    Returns
    -------
    ingap : float
        how much of the band is in any gap
    """
    ingap = []
    for gap in gaps:
        ingap.append(float(len(np.where(np.logical_and(band > gap[0], band < gap[1]))[0])) / float(len(gap)))

    return np.array(ingap)


def estimate_gaps_from_bands(bands, gapthres, de):
    """

    Returns
    -------

    """
    erange = np.arange(np.min(bands.ravel()), np.max(bands.ravel()), de)
    # Get gaps where DOS falls below thres
    ii = 0
    density = np.array([len(np.unique(np.where(np.logical_and(bands > erange[ii], bands < erange[ii + 1]))[0]))
                        for ii in range(len(erange) - 1)])

    # Check
    # plt.plot(np.arange(len(density)), density, 'b.-')
    # plt.show()

    # Look for contiguous energy ranges
    wheregap = np.where(density < gapthres)[0]
    edgemarkers = np.where(np.abs(np.diff(wheregap)) > 1)[0]
    print 'edgemarkers = ', edgemarkers
    print 'erange[wheregap[edgemarkers]] = ', erange[wheregap[edgemarkers]]
    inds = wheregap[edgemarkers].tolist() + (wheregap[edgemarkers + 1]).tolist()
    inds.append(wheregap[0])
    inds.append(wheregap[len(wheregap) - 1])
    print 'inds = ', inds

    inds = np.sort(np.array(inds).ravel())
    print 'inds = ', inds
    gaps = (erange[inds]).reshape(2, -1)
    gaps[:, 1] += de
    print 'gaps = ', gaps

    return gaps


if __name__ == '__main__':
    gap_finder = False
    color_disp = True
    if gap_finder:
        NN = 1000
        bands = np.random.rand(NN)
        bands[bands > 0.5] += 0.2
        bands[bands > 1.0] += 0.5
        plt.hist(bands, bins=100)
        gapthres = 2
        de = 0.02
        gaps = estimate_gaps_from_bands(bands, gapthres, de)
        for gap in gaps:
            plt.plot([gap[0], gap[0]], [0, np.sqrt(NN)], 'r--')
            plt.plot([gap[1], gap[1]], [0, np.sqrt(NN)], 'g--')

        print 'gaps = ', gaps
        plt.show()

    if color_disp:
        import pickle
        import glob

        geom = 'kagome'
        fnbase = '/Users/npmitchell/Dropbox/Soft_Matter/PAPER/gyro_disorder_paper/figure_drafts/' + \
                 'data_for_figs/kagome_fig/' + geom + '/'
        print 'loading kspace chern from: ' + fnbase + '*strip_dispserion.pkl'
        globfn = glob.glob(fnbase + '*strip_dispersion.pkl')
        print 'globfn = ', globfn
        with open(globfn[0], "rb") as fn:
            dat_dict = pickle.load(fn)

        # Unpack the dictionary
        kx = dat_dict['kx']
        bands = dat_dict['bands']

        # Keep only the bands above omega = 0
        bands = np.array(bands)
        # bands_out = None
        # for band in bands:
        #     if bands_out is None:
        #         bands_out = band
        #     else:
        #         bands_out = np.dstack()
        # keep = []
        # for band in bands:
        #     if (band > 0).all():
        #         keep.append(ii)

        keep = np.unique(np.where((bands > 0))[1])
        bands = bands[:, keep]

        # Plot the dispersion
        stripax = periodicstrip(kx, bands, ax=None, colorgaps=True, de=0.05)
        plt.show()
        sys.exit()

