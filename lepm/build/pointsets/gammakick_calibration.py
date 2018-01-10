import numpy as np
import lepm.dataio as dio
import lepm.plotting.colormaps as lecmaps
import lepm.stringformat as sf
import lepm.plotting.plotting as leplt
import matplotlib.pyplot as plt
import lepm.build.pointsets.gammakick as gammakick
import cPickle as pkl
import glob
import scipy.interpolate

'''Functions used in running gammakick.py with calibration arguments.
These are used to measure the powerlaw kick amplitudes of points perturbed from a disordered square lattice.
'''


def calibrate_kicksz_epsilon(gammav, epsilonv, nn, rootdir='./'):
    """

    Parameters
    ----------
    gammav
    epsilonv
    nn
    rootdir

    Returns
    -------

    """
    # To quantify the distance use the expectation of the log of zeta, as discussed here
    # http://stats.stackexchange.com/questions/121925/intuitive-descriptive-statistics-for-power-law-distributions
    # gammav = np.arange(0.1, 2.1, 0.02)
    # kickszv = np.logspace(-18, 3, 66)
    # nn = args.N

    # Prepare output dirs
    seriesname = 'kicksz_distance/nn' + str(nn) + '_ngammav' + str(len(gammav)) + '_nkickszv' + str(len(kickszv))
    caloutdir = rootdir + 'calibration/' + seriesname + '/'
    datoutdir = rootdir + 'calibration_data/' + seriesname + '/'
    dio.ensure_dir(caloutdir)
    dio.ensure_dir(datoutdir)

    # Set up plot
    fig, ax = leplt.initialize_1panel_fig(wsfrac=0.5, hsfrac=0.35, y0frac=0.14, x0frac=0.18)
    ax.set_xlabel(r'transformation parameter, $\log_{10} \epsilon$')
    ax.set_ylabel(r'expected kick size, $\frac{1}{N} \sum_i \log \zeta_i$')
    plt.suptitle('Kick size calibration')

    # # Set up linear function for curve fitting
    # def linear(x, a, b):
    #     return a * x + b

    colorv = lecmaps.husl_palette(len(gammav) + 2, h=0.02, l=0.5)
    # Check if data already saved
    if glob.glob(datoutdir + 'datadict.pkl'):
        with open(datoutdir + 'datadict.pkl', "r") as f:
            datadict = pkl.load(f)
        datafits = np.loadtxt(datoutdir + 'datafits.txt', unpack=False)

        ii = 0
        for gamma in gammav:
            gammastr = sf.float2pstr(gamma)
            plt.plot(np.log10(epsilonv), datadict[gammastr][:, 1], '.-', color=colorv[ii],
                     label=r'$\gamma = $' + '{0:0.2f}'.format(gamma), markersize=2)
            ii += 1

    else:
        # Initialize data containers: a dict and an array
        datadict = {'readme': 'Each gamma key is an n x 3 array, where the columns are kicksz, ' +
                              'average log value np.sum(np.log10(zeta))/float(nn), ' +
                              'standard deviation of the log values ' +
                              'np.sqrt(np.sum((np.log10(zeta) - avglog)**2) / float(nn)),' +
                              'and minimum kick np.min(zeta). nn is the number of kicks (particles).' +
                              'zeta are the individual kicks.'}
        datafits = np.zeros((len(gammav), 5), dtype=float)
        datafits[:, 0] = gammav

        ii = 0
        for gamma in gammav:
            gammastr = sf.float2pstr(gamma)
            datadict[gammastr] = np.zeros((len(epsilonv), 4), dtype=float)
            datadict[gammastr][:, 0] = epsilonv
            kk = 0
            for eps in epsilonv:
                epsstr = '{0:0.2e}'.format(eps)
                zeta = gammakick.gammapowerlaw_random(gamma, eps, size=nn)
                avglog = np.sum(np.log10(zeta)) / float(nn)
                stdlog = np.sqrt(np.sum((np.log10(zeta) - avglog) ** 2) / float(nn))
                minz = np.min(zeta)
                datadict[sf.float2pstr(gamma)][kk, 1:4] = np.array([avglog, stdlog, np.min(zeta)])
                kk += 1

            # Fit the result to a line
            avglogv = datadict[sf.float2pstr(gamma)][:, 1]
            stdlogv = datadict[sf.float2pstr(gamma)][:, 2]
            # popt, pcov = curve_fit(linear, kickszv, avglogv, sigma=None)
            popt, pcov = np.polyfit(np.log10(epsilonv), avglogv, 1, rcond=None, full=False, w=None, cov=True)
            datafits[ii, 1:] = np.array([popt[0], popt[1], np.sqrt(pcov[0, 0]), np.sqrt(pcov[1, 1])])

            # Add data to current plot
            plt.plot(np.log10(epsilonv), datadict[gammastr][:, 1], '.-', color=colorv[ii],
                     label=r'$\gamma = $' + '{0:0.2f}'.format(gamma), markersize=2)
            # # check fit
            # plt.plot(np.log10(kickszv), popt[0] * np.log10(kickszv) + popt[1], 'k-')
            # plt.show()
            # sys.exit()
            # plt.semilogx(kickszv, datadict[gammastr][:, 2], '.-', label=r'$\min(\zeta)$')
            ii += 1

    # Save data
    with open(datoutdir + 'datadict.pkl', "w") as f:
        pkl.dump(datadict, f)
    header = 'Fitting the dependence of the size of a kick (avg of log_10 of kicksz) on the ' + \
             'parameter epsilon and exponent gamma. The slope is of avglog zeta versus log_10(epsilon). ' + \
             'Values here are: gamma fit_slope fit_intercept unc_slope unc_intercept'
    np.savetxt(datoutdir + 'datafits.txt', datafits, header=header)

    # Finish the plot
    if len(gammav) < 13:
        ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
    plt.savefig(caloutdir + 'cal_kicksz_epsilon.png', dpi=200)
    plt.close('all')

    # Look at curve fits
    fig, ax = leplt.initialize_1panel_fig(wsfrac=0.5, hsfrac=0.35, y0frac=0.14, x0frac=0.25)
    plt.errorbar(gammav, datafits[:, 1], yerr=datafits[:, 3], fmt='.')
    ax.set_xlabel(r'$S(k)$ exponent, $\gamma$')
    ax.set_ylabel(r'$\partial_{\log_{10} \epsilon} \left[\frac{1}{N} \sum_i \log_{10} \zeta_i\right] $')
    plt.suptitle(r'Kick size calibration: kick size dependence on $\gamma$')
    plt.savefig(caloutdir + 'cal_dkickzdeps_gamma.png', dpi=200)
    plt.close('all')

    # Look at curve fit intercept
    fig, ax = leplt.initialize_1panel_fig(wsfrac=0.5, hsfrac=0.35, y0frac=0.14, x0frac=0.25)
    plt.errorbar(gammav, datafits[:, 2], yerr=datafits[:, 4], fmt='.')
    ax.set_xlabel(r'$S(k)$ exponent, $\gamma$')
    ax.set_ylabel(r'$\left.\frac{1}{N} \sum_i \log_{10} \zeta_i\right|_{\epsilon = 1}$')
    plt.suptitle(r'Kick size calibration: kick size dependence on $\gamma$')
    plt.savefig(caloutdir + 'cal_kickzintercept_gamma.png', dpi=200)
    plt.close('all')

    # Look at curve fit intercept
    fig, ax = leplt.initialize_1panel_fig(wsfrac=0.5, hsfrac=0.35, y0frac=0.14, x0frac=0.25)
    plt.errorbar(np.log10(gammav), datafits[:, 2], yerr=datafits[:, 4], fmt='.')
    ax.set_xlabel(r'$S(k)$ exponent, $\log_{10}\gamma$')
    ax.set_ylabel(r'$\left.\frac{1}{N} \sum_i \log_{10} \zeta_i\right|_{\epsilon = 1}$')
    plt.suptitle(r'Kick size calibration: kick size dependence on $\gamma$')
    plt.savefig(caloutdir + 'cal_kickzintercept_loggamma.png', dpi=200)
    plt.close('all')


def retrieve_epsilon(kicksz, gamma, calibdata_dir='auto'):
    """Load the calibration data relating kicksize (average log of kick magnitudes) and gamma (hyperuniform exponent)
    to the appropriate epsilon.

    Parameters
    ----------
    kicksz
    gamma
    calibdata_dir

    Returns
    -------

    """
    # Given gamma, given kicksz --> gives epsilon for transformation
    # Using the data from nn100000_ngammav100_nkickszv66, invert to find epsilon as a function of kicksz and gamma.
    # Choose a kick size, convert to epsilon, and test that choice of epsilon for the desired S(k) scaling as k->0,
    # for different values of gamma.
    if calibdata_dir == 'auto':
        calibdata_dir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/networks/random_organization_source/' +\
                        'random_kick_gamma_npm/calibration_data/kicksz_epsilon/nn1000000_ngammav100_nkickszv66/'

    gammav, slope, intcp, unc_s, unc_i = np.loadtxt(calibdata_dir + 'datafits.txt', unpack=True)
    # Invert the curves. There is one curve per value of gamma
    # log10(eps) = 1/m * log10zeta - b/m, where m and b are slope and intrc.
    # The slope is equal to unity, so just set that equal to one below.
    # Interpolate slope and intercept as function of gamma
    # slopef = scipy.interpolate.interp1d(gammav, slope)
    intcpf = scipy.interpolate.interp1d(gammav, intcp)
    # mm = slopef(gamma)
    bb = intcpf(gamma)
    eps = 10**(kicksz - bb)

    # check
    # plt.plot(gammav, slope)
    # plt.plot(gammav, intcp)
    # plt.show()
    # sys.exit()

    return eps
