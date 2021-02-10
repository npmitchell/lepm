from scipy import interpolate
import numpy as np
from scipy import optimize
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
import lepm.plotting.plotting as leplt
import sys


'''Functions for collapsing many curves of data.
'''


############################################################
# The finite size transformation and related transformations
############################################################
def fermi_function(ee, ef, T):
    return 1. / (np.exp((ee - ef) / T) + 1)


def residual(p, fitfunc, x, y, yerr):
    """Compute residuals of a fitfunction with parameters p.
    If yerr has elements which are zero, compute the non-error-weighted residuals

    Parameters
    ----------
    p : list of floats
        fitting parameters
    fitfunc : function
        the function which takes p and x as input
    x : array or list
        the x coordinate data
    y : array or list
        the y coordinate data
    yerr : float array
        the uncertainties in the data
    """
    if (yerr == 0.).any():
        return fitfunc(p, x) - y
    return (fitfunc(p, x) - y) / yerr


def finsz_fixed_xc(nu, xx, sz, sz0, xc):
    """Rescaling for single parameter finite size effect analysis: Given a list of float arrays xx or 2D float array xx,
    rescale each row (array) of xx by finite size to a power 1/nu.

    Parameters
    ----------
    nu : float
        The power for rescaling by system size
    xx : list of N float arrays (of possible variable length) or N x M float array
        The parameter that is varying across the phase boundary, spanning the domain of the data for each curve
    sz : list or array of floats
        The size of the system, to govern finite size scaling
    sz0 : float
        The reference size of the system, best picked to be an intermediate size
    xc : float
        The value of the parameter x at the phase boundary
    """
    out = []
    for ii in range(len(xx)):
        out.append(finsz_xc_transf(nu, xx[ii], sz[ii], sz0, xc))
    return out


def finsz_xc_transf(nu, xx, sz, sz0, xc):
    """

    Parameters
    ----------
    nu : float
        The power for rescaling by system size
    xx : M x 1 float array
        The parameter that is varying across the phase boundary, spanning the domain of the data for each curve
    sz : float
        The size of the system, to govern finite size scaling
    sz0 : float
        The reference system size, best picked to be an intermediate size
    xc : float
        critical value of x for the phase boundary

    Returns
    -------
    """
    return xc + (xx - xc) * (float(sz) / float(sz0)) ** (1./float(nu))


#############################################################################
# High-level functions for getting bounds and minimizing variance in curves
#############################################################################
def get_bounds_finsz_fixed_xc(xx, xmin, xmax, sz, sz0, xc, check=False):
    """Given the ranges of data in xx (either a list of arrays of a 2D array). Run this before a constrained
    optimization to get the constraint.
    Note: It is necessary that np.min(xx[ii]) < xmin for all ii in range(len(xx)), and similarly for the max

    Parameters
    ----------
    xx : list of N float arrays (of possible variable length) or N x M float array
        The parameter that is varying across the phase boundary, spanning the domain of the data for each curve
    xmin : float
        The lower bound of the domain of the phase boundary considered
    xmax : float
        The upper bound of the domain of the phase boundary considered
    transf : str
        A string specifier for the function that transforms data
    transf_args : tuple, additional args for supplied function transf
        The additional arguments for the supplied function transf, after nu (which is the var to be bounded) and xx.
        If transf==finsz_fixed_xc, then additional args are (sz, sz0, xc).

    Returns
    -------
    (numin, numax) : tuple of floats
        lower and upper limits for the bounds of the optimization, which put the furthest data at the tails of the
        curves to be fitted right at the xmin or xmax of the range considered.
    """
    # Solve for min and max allowed nu such that end of data reaches bounds of domain
    sz0 = float(sz0)
    ii = 0
    numin = None
    numax = None
    minfirst = True
    maxfirst = True
    for xii in xx:
        xlo = np.min(xii)
        xhi = np.max(xii)
        if sz[ii] != sz0:
            # Since x' = xc + (x - xc) * (L/L0)**(1/nu), define gamma == 1/nu. Then, note we need
            # (xmax - xc)/(x' - xc) < 1
            # (xmax - xc)/(xc - xc + (xendpt - xc) * (L/L0)**gamma) < 1
            # (xmax - xc)/(x_endpt - xc) < (L/L0)**(gamma)
            # ln(x.../x...) < ln(L/L0) * gamma  (and similarly with xmin)
            # Now be careful since the ln(L/L0) could be negative.
            # The ln(x.../x...) term is ALWAYS negative, since the endpts are more distant from xc.
            # Let c1 = ln(L/L0) / ln((xmax - xc)/(x1 - xc)),
            #     c2 = ln(L/L0) / ln((xmin - xc)/(x2 - xc))
            # Case 1: L<L0, so ln(L/L0) < 0, so c1 and c2 are positive. Assume nu is positive. Then
            #         gamma = 1/nu < 1/c1, 1/c2
            #         nu > c1, c2
            # Case 2: L>L0, so ln(L/L0) > 0, so c1 and c2 are negative. Assume nu > 0, so
            #         gamma = 1/nu > 1/c1, 1/c2
            #         nu > c1, c2 --> here the only assumption is nu > 0
            #         Interpretation: a number greater than 1 (ie L/L0) raised to any positive power gives
            #           can only stretch the distance of an endpt from xc.
            #           Limits: nu -> 0, maximal stretching
            #                   nu -> infty, no stretching (but no shrinking either!)
            #
            # For L/L0 < 1, then nu > max(c1, c2)
            # For L/L0 > 1, then nu > max(c1, c2) which are below zero
            c1 = np.log(float(sz[ii]) / sz0) / np.log((xmax - xc) / (xhi - xc))
            c2 = np.log(float(sz[ii]) / sz0) / np.log((xc - xmin) / (xc - xlo))
            if sz[ii] < sz0:
                numin_ii = max(c1, c2)
                if check:
                    print 'c1 = ', c1
                    print 'c2 = ', c2
                    print 'sz < sz0: numin_ii =', numin_ii
                    print 'power = {0:0.5f}'.format((float(sz0) / sz[ii]) ** numin_ii)
                    print 'xprime_lo =', xc + (xlo - xc) * (sz0 / float(sz[ii])) ** numin_ii
                    print 'xprime_hi =', xc + (xhi - xc) * (sz0 / float(sz[ii])) ** numin_ii
                    print 'numin xprime_lo === >'
                    print finsz_xc_transf(numin_ii, xlo, sz[ii], sz0, xc)
                    print 'numin xprime_hi === >'
                    print finsz_xc_transf(numin_ii, xhi, sz[ii], sz0, xc)

                if minfirst:
                    numin = numin_ii
                    minfirst = False
                else:
                    print 'numin = ', numin
                    print 'numin_ii = ', numin_ii
                    numin = np.min(np.array([numin, numin_ii]))
            # else:
            #     pass
            #     # numax_ii = min(c1, c2)
            #     # print 'c1 = ', c1
            #     # print 'c2 = ', c2
            #     # print 'sz > sz0: numax_ii =', numax_ii
            #     # print 'power = ', (sz[ii] / float(sz0)) ** (1. / numax_ii)
            #     # print 'sz < sz0: numax_ii =', numax_ii
            #     # print 'power = {0:0.5f}'.format((float(sz0) / sz[ii]) ** numax_ii)
            #     # print 'xprime_lo =', xc + (xlo - xc) * (sz0 / float(sz[ii])) ** numax_ii
            #     # print 'xprime_hi =', xc + (xhi - xc) * (sz0 / float(sz[ii])) ** numax_ii
            #     # print 'numax xprime_lo === >'
            #     # print finsz_xc_transf(numax_ii, xlo, sz[ii], sz0, xc)
            #     # print 'numax xprime_hi === >'
            #     # print finsz_xc_transf(numax_ii, xhi, sz[ii], sz0, xc)
            #     # if maxfirst:
            #     #     numax = numax_ii
            #     #     minfirst = False
            #     # else:
            #     #     numax = np.max(numax, numax_ii)

        ii += 1

    return numin, numax


def minimize_variance_curves_fixed_xc(p0, xx, yy, sz, sz0, xmin, xmax, xc, apply_bounds=True, order=1, check=False,
                                      view=False, pausetime=0.2, mindx_frac=0.05, maxdx_frac=0.2):
    """Optimize finite scaling formula for known critical pt

    Parameters
    ----------
    p0
    xx
    yy
    sz
    sz0
    xmin :
        minimum value of x over which to compute variance in curves, and also over which to
        examine curvature of sum of squared residuals
    xmax :
        maximum value of x over which to compute variance in curves, and also over which to
        examine curvature of sum of squared residuals
    xc
    bounds : sequence, optional
        Bounds for variables: (min, max) pairs for each element in x, defining the bounds on that parameter.
        Use None for one of min or max when there is no bound in that direction.
    mindx_frac : float
    maxdx_frac : float
        maximum range of interval (xmin, xmax) on which to fit residuals the quadratic....?

    Returns
    -------
    nustar, uncertainty
    """
    if apply_bounds:
        # Decide on bounds for minimization using the finite size scaling formula to get the bounds of the fit
        bounds = get_bounds_finsz_fixed_xc(xx, xmin, xmax, sz, sz0, xc, check=check)
        print 'bounds = ', bounds
        bounds = (bounds[0] + 1e-7 * bounds[0], bounds[1])

        if check or view:
            plt.clf()
            # Check out initial data
            colors = plt.get_cmap('copper')(np.linspace(0., 1., len(xx)))
            for ii in range(len(xx)):
                plt.plot(xx[ii], yy[ii], '-', color=colors[ii], label='sz = ' + str(sz[ii]))

            plt.title('Initial uncollapsed data')
            plt.legend()
            if check:
                plt.show()
            else:
                plt.pause(pausetime)
            plt.clf()

            # Show result of rescaling by nu in bounds of nu
            xrescale = finsz_fixed_xc(bounds[0], xx, sz, sz0, xc)
            for ii in range(len(xx)):
                plt.plot(xrescale[ii], yy[ii], '-', color=colors[ii], label='sz = ' + str(sz[ii]))
            plt.title(r'Lower bound for $x \in ($' + '{0:0.3f}'.format(xmin) + ',' +
                      '{0:0.3f}'.format(xmax) + r'$)$: $\nu = $' + str(bounds[0]))
            plt.legend()
            if check:
                plt.show()
            else:
                plt.pause(pausetime)
            plt.clf()
            if bounds[1] is not None:
                xrescale = finsz_fixed_xc(bounds[1], xx, sz, sz0, xc)
                for ii in range(len(xx)):
                    plt.plot(xrescale[ii], yy[ii], '-', color=colors[ii], label='sz = ' + str(sz[ii]))
                plt.title(r'Upper bound for $x \in ($' + '{0:0.3f}'.format(xmin) + ',' +
                          '{0:0.3f}'.format(xmax) + r'$)$: $\nu = $' + str(bounds[1]))
                plt.legend()
                if check:
                    plt.show()
                else:
                    plt.pause(pausetime)
                plt.clf()
    else:
        bounds = (None, None)

    result = minimize(variance_finsz_fixed_xc, p0, args=(xx, yy, sz, sz0, xmin, xmax, xc, order, False),
                      method='L-BFGS-B', bounds=(bounds,), options={'disp': False})

    # Save resulting optimized value of nu
    nustar = result.x[0]

    # Estimate error
    # Resulting minimized error
    xrescale = finsz_fixed_xc(nustar, xx, sz, sz0, xc)
    var_min = variance_curves(xrescale, yy, xmin, xmax, order=1, check=False)
    uncs, errs = nu_error_asymptotic_parabolas(xx, yy, sz, sz0, xc, nustar, xmin, xmax, mindx_frac, maxdx_frac, 10,
                                               var_min, check=check, view=view)

    # print 'errs = ', errs
    # sys.exit()
    # # Pick uncertainty with lowest covariance (ie the one that fits the curve best)
    # unc = uncs[np.argmin(errs)]
    # Weight the uncertainty by the errors
    unc = np.sum(uncs / errs) / np.sum(1. / errs)

    # Fit uncertainties to a line and extrapolate to zero
    # pp = np.polyfit(np.linspace(mindx_frac, maxdx_frac, 10), uncs, 1)
    # unc = pp[1]

    if check:
        xrescale = finsz_fixed_xc(nustar, xx, sz, sz0, xc)
        for ii in range(len(xx)):
            plt.plot(xrescale[ii], yy[ii], '-', color=colors[ii], label='sz = ' + str(sz[ii]))
        ylims = plt.gca().get_ylim()
        plt.plot([xmin, xmin], [ylims[0], ylims[1]], 'k--')
        plt.plot([xmax, xmax], [ylims[0], ylims[1]], 'k--')
        print 'xc = ', xc
        print 'nustar = ', nustar
        plt.title('This is the result of optimization for ' + r'$\nu$ with fixed $x_c$: ' +
                  r'$\nu = $' + '{0:0.3f} for '.format(nustar) +
                  r'$x_c =$' + '{0:0.3f}'.format(xc))
        plt.legend()
        plt.show()

    return nustar, unc


def minimize_variance_curves_fixed_nu(p0, xx, yy, sz, sz0, xmin, xmax, nu, apply_bounds=True, order=1, check=False,
                                      view=False, pausetime=0.2, mindx_frac=0.05, maxdx_frac=0.2):
    """Optimize finite scaling formula for known critical exponent

    Parameters
    ----------
    p0
    xx
    yy
    sz
    sz0
    xmin :
        minimum value of x over which to compute variance in curves, and also over which to
        examine curvature of sum of squared residuals
    xmax :
        maximum value of x over which to compute variance in curves, and also over which to
        examine curvature of sum of squared residuals
    xc
    bounds : sequence, optional
        Bounds for variables: (min, max) pairs for each element in x, defining the bounds on that parameter.
        Use None for one of min or max when there is no bound in that direction.
    mindx_frac : float
    maxdx_frac : float
        maximum range of interval (xmin, xmax) on which to fit residuals the quadratic....?

    Returns
    -------
    xcstar, uncertainty
    """
    if apply_bounds:
        # Decide on bounds for minimization using the finite size scaling formula to get the bounds of the fit
        bounds = get_bounds_finsz_fixed_xc(xx, xmin, xmax, sz, sz0, p0[0], check=check)
        print 'bounds = ', bounds
        bounds = (bounds[0] + 1e-7 * bounds[0], bounds[1])

        if check or view:
            plt.clf()
            # Check out initial data
            colors = plt.get_cmap('copper')(np.linspace(0., 1., len(xx)))
            for ii in range(len(xx)):
                plt.plot(xx[ii], yy[ii], '-', color=colors[ii], label='sz = ' + str(sz[ii]))

            plt.title('Initial uncollapsed data')
            plt.legend()
            if check:
                plt.show()
            else:
                plt.pause(pausetime)
            plt.clf()

            # Show result of rescaling by nu in bounds of nu
            xrescale = finsz_fixed_xc(bounds[0], xx, sz, sz0, xc)
            for ii in range(len(xx)):
                plt.plot(xrescale[ii], yy[ii], '-', color=colors[ii], label='sz = ' + str(sz[ii]))
            plt.title(r'Lower bound for $x \in ($' + '{0:0.3f}'.format(xmin) + ',' +
                      '{0:0.3f}'.format(xmax) + r'$)$: $\nu = $' + str(bounds[0]))
            plt.legend()
            if check:
                plt.show()
            else:
                plt.pause(pausetime)
            plt.clf()
            if bounds[1] is not None:
                xrescale = finsz_fixed_xc(bounds[1], xx, sz, sz0, xc)
                for ii in range(len(xx)):
                    plt.plot(xrescale[ii], yy[ii], '-', color=colors[ii], label='sz = ' + str(sz[ii]))
                plt.title(r'Upper bound for $x \in ($' + '{0:0.3f}'.format(xmin) + ',' +
                          '{0:0.3f}'.format(xmax) + r'$)$: $\nu = $' + str(bounds[1]))
                plt.legend()
                if check:
                    plt.show()
                else:
                    plt.pause(pausetime)
                plt.clf()
    else:
        bounds = (None, None)

    result = minimize(variance_finsz_fixed_nu, p0, args=(xx, yy, sz, sz0, xmin, xmax, nu, order, False),
                      method='L-BFGS-B', bounds=(bounds,), options={'disp': False})

    # Save resulting optimized value of nu
    xcstar = result.x[0]

    # Estimate error
    # Resulting minimized error
    xrescale = finsz_fixed_xc(nu, xx, sz, sz0, xcstar)
    var_min = variance_curves(xrescale, yy, xmin, xmax, order=1, check=False)
    uncs, errs = xc_error_asymptotic_parabolas(xx, yy, sz, sz0, xcstar, nu, xmin, xmax, mindx_frac, maxdx_frac, 10,
                                               var_min, check=check, view=view)

    # print 'errs = ', errs
    # sys.exit()
    # # Pick uncertainty with lowest covariance (ie the one that fits the curve best)
    # unc = uncs[np.argmin(errs)]
    # Weight the uncertainty by the errors
    print('collapse_curves: uncertainties = ', uncs)
    unc = np.sum(uncs / errs) / np.sum(1. / errs)

    # Fit uncertainties to a line and extrapolate to zero
    # pp = np.polyfit(np.linspace(mindx_frac, maxdx_frac, 10), uncs, 1)
    # unc = pp[1]

    if check:
        xrescale = finsz_fixed_xc(nu, xx, sz, sz0, xcstar)
        for ii in range(len(xx)):
            plt.plot(xrescale[ii], yy[ii], '-', color=colors[ii], label='sz = ' + str(sz[ii]))
        ylims = plt.gca().get_ylim()
        plt.plot([xmin, xmin], [ylims[0], ylims[1]], 'k--')
        plt.plot([xmax, xmax], [ylims[0], ylims[1]], 'k--')
        print 'xc = ', xc
        print 'nustar = ', nustar
        plt.title('This is the result of optimization for ' + r'$\nu$ with fixed $x_c$: ' +
                  r'$\nu = $' + '{0:0.3f} for '.format(nustar) +
                  r'$x_c =$' + '{0:0.3f}'.format(xc))
        plt.legend()
        plt.show()

    return xcstar, unc


def minimize_variance_curves(p0, xx, yy, sz, sz0, deltax, apply_bounds=True, xcbounds=None,
                             order=1, check=False, view=False, mindx_frac=0.05, maxdx_frac=0.2,
                             mindnu_frac=0.05, maxdnu_frac=0.2):
    """Optimize finite scaling formula for known critical pt and transition position. Here both the transition
    coordinate xc and the scaling exponent nu are optimized. Applying bounds uses the initial guess for xc to determine
    the limits of the optimization. If an error is returned because the optimal nustar was on a bounded value, then
    (1) try again with better initial guess for xc, or (2) Get more data in the tails of the curves, far from xc.

    Parameters
    ----------
    p0 : list of two floats
        The initial guesses for xc, nu
    xx : list of N float arrays (of possible variable length) or N x M float array
        The parameter that is varying across the phase boundary, spanning the domain of the data for each curve
    yy : list of N float arrays (of possible variable length) or N x M float array
        y coords for N curves with M points each (possibly variable #pts in each curve)
    sz : list of floats
        The list of system sizes
    sz0 : float
        The reference system size which is unaffected by the scaling
    deltax : float
        Width of the optimization window around xc -- ie optimization window is (xc - delta * 0.5, xc + delta * 0.5).
        If apply_bounds is true and xcbounds is given, then deltax acts as the window over which to find the radius of
        curvature of the sum of squared residuals to get an error measurement for the fit.
    apply_bounds : bool
        Whether to apply boundaries from limits of data to the fitting parameters (in particular, to nu, the scaling
        exponent)
    xcbounds : tuple of two floats or None
        If apply_bounds is True, xcbounds gives extremal values for xcbounds rather than using the default of
        (p0[0] - deltax * 0.5, p0[0] + deltax * 0.5)
    bounds : sequence, optional
        Bounds for variables: (min, max) pairs for each element in x, defining the bounds on that parameter.
        Use None for one of min or max when there is no bound in that direction.

    Returns
    -------
    nustar, uncertainty
    """
    view = True
    # guess the max and min of the range that is to be collapsed
    xmin_guess = p0[0] - deltax * 0.5
    xmax_guess = p0[0] + deltax * 0.5
    if apply_bounds:
        # Decide on bounds for minimization using the finite size scaling formula to get the bounds of the fit
        bounds = get_bounds_finsz_fixed_xc(xx, xmin_guess, xmax_guess, sz, sz0, p0[0], check=check)
        print 'bounds = ', bounds
        bounds = (bounds[0] * (1+ 1e-7), bounds[1])

        if check:
            # Check out initial data
            colors = plt.get_cmap('copper')(np.linspace(0., 1., len(xx)))
            for ii in range(len(xx)):
                plt.plot(xx[ii], yy[ii], '-', color=colors[ii], label='sz = ' + str(sz[ii]))

            plt.title('Initial uncollapsed data')
            plt.legend()
            plt.show()

            # Show bounds
            xrescale = finsz_fixed_xc(bounds[0], xx, sz, sz0, p0[0])
            for ii in range(len(xx)):
                plt.plot(xrescale[ii], yy[ii], '-', color=colors[ii], label='sz = ' + str(sz[ii]))
            ylims = plt.gca().get_ylim()
            plt.plot([xmin_guess, xmin_guess], [ylims[0], ylims[1]], 'k--')
            plt.plot([xmax_guess, xmax_guess], [ylims[0], ylims[1]], 'k--')
            plt.title('This is the result of optimization after \napplying the lower bound for ' + r'$\nu$: ' +
                      r'$\nu = $' + '{0:0.3f} for '.format(bounds[0]) +
                      r'$x \in ($' + '{0:0.3f}'.format(xmin_guess) + ',' +
                      '{0:0.3f}'.format(xmax_guess) + r'$)$')
            plt.legend()
            plt.show()
            if bounds[1] is not None:
                xrescale = finsz_fixed_xc(bounds[1], xx, sz, sz0, xc)
                for ii in range(len(xx)):
                    plt.plot(xrescale[ii], yy[ii], '-', color=colors[ii], label='sz = ' + str(sz[ii]))
                ylims = plt.gca().get_ylim()
                plt.plot([xmin_guess, xmin_guess], [ylims[0], ylims[1]], 'k--')
                plt.plot([xmax_guess, xmax_guess], [ylims[0], ylims[1]], 'k--')
                plt.title(r'Upper bound for $x \in ($' + '{0:0.3f}'.format(xmin) + ',' +
                          '{0:0.3f}'.format(xmax) + r'$)$: $\nu = $' + str(bounds[1]))
                plt.legend()
                plt.show()
        if xcbounds is None:
            xcbounds = (xmin_guess, xmax_guess)
    else:
        bounds = (None, None)
        xcbounds = (None, None)

    result = minimize(variance_finsz, p0, args=(xx, yy, sz, sz0, deltax, order, False),
                      method='L-BFGS-B', bounds=(xcbounds, bounds,), options={'disp': True})

    # Save resulting minimum value of nu
    xcstar, nustar = result.x
    xmin = xcstar - deltax * 0.5
    xmax = xcstar + deltax * 0.5

    # Estimate error
    # Resulting minimized error in nu
    xrescale = finsz_fixed_xc(nustar, xx, sz, sz0, xcstar)
    var_min = variance_curves(xrescale, yy, xmin, xmax, order=1, check=False)
    nu_uncs, nu_errs = nu_error_asymptotic_parabolas(xx, yy, sz, sz0, xcstar, nustar, xmin, xmax, mindnu_frac,
                                                     maxdnu_frac, 10, var_min, check=check, view=view)

    xc_uncs, xc_errs = xc_error_asymptotic_parabolas(xx, yy, sz, sz0, xcstar, nustar, xmin, xmax, mindxc_frac,
                                                     maxdxc_frac, 10, var_min, check=check, view=view)

    # Weight the uncertainty by the errors
    # unc = uncs[np.argmin(errs)]
    unc_nu = np.sum(nu_uncs / nu_errs) / np.sum(1. / nu_errs)
    unc_xc = np.sum(xc_uncs / xc_errs) / np.sum(1. / xc_errs)

    # Fit uncertainties to a line and extrapolate to zero
    # pp = np.polyfit(np.linspace(mindx_frac, maxdx_frac, 10), uncs, 1)
    # unc = pp[1]

    if check:
        xrescale = finsz_fixed_xc(nustar, xx, sz, sz0, xcstar)
        for ii in range(len(xx)):
            plt.plot(xrescale[ii], yy[ii], '-', color=colors[ii], label='sz = ' + str(sz[ii]))
        ylims = plt.gca().get_ylim()
        plt.plot([xmin_guess, xmin_guess], [ylims[0], ylims[1]], 'k--')
        plt.plot([xmax_guess, xmax_guess], [ylims[0], ylims[1]], 'k--')
        plt.title('This is the result of optimization for ' + r'$x_c, \nu$: ' +
                  r'$\nu = $' + '{0:0.3f} for '.format(nustar) +
                  r'$x_c =$' + '{0:0.3f}'.format(xcstar) + r'$)$')
        plt.legend()
        plt.show()

    return xcstar, nustar, unc_nu, unc_xc


############################################################
# The error function to minimize: integral of standard dev
############################################################
def variance_finsz_fixed_xc(pp, xx, yy, sz, sz0, xmin, xmax, xc, order, check):
    xrescale = finsz_fixed_xc(pp[0], xx, sz, sz0, xc)
    res = variance_curves(xrescale, yy, xmin, xmax, order=order, check=check)
    return res


def variance_finsz_fixed_nu(pp, xx, yy, sz, sz0, xmin, xmax, nu, order, check):
    xrescale = finsz_fixed_xc(nu, xx, sz, sz0, pp[0])
    res = variance_curves(xrescale, yy, xmin, xmax, order=order, check=check)
    return res


def variance_finsz(pp, xx, yy, sz, sz0, deltax, order, check):
    """

    Parameters
    ----------
    pp : list of two floats
        The (current) values for xc, nu: the center of the transition and the system size scaling exponent
    xx
    yy
    sz
    sz0
    deltax
    order
    check

    Returns
    -------

    """
    xrescale = finsz_fixed_xc(pp[1], xx, sz, sz0, pp[0])
    res = variance_curves(xrescale, yy, pp[0] - deltax * 0.5, pp[0] + deltax * 0.5, order=order, check=check)
    return res


def variance_curves(xx, yy, xmin, xmax, order=1, check=False, nsamples=100):
    """Sum the variance of interpolated curves (xx[ii, yy[ii]) over a sampling from xmin to xmax, for each curve
    (xx[ii], yy[ii]) in xx and yy.
    Note that yy[ii] must be a (single-valued) function of xx[ii] for this to be useful.

    Parameters
    ----------
    xx : list of N float arrays (of possible variable length) or N x M float array
        The parameter that is varying across the phase boundary, spanning the domain of the data for each curve
    yy : list of N float arrays (of possible variable length) or N x M float array
        y coords for N curves with M points each (possibly variable #pts in each curve)
    xmin : float
        The lower bound for the domain that is to be collapsed
    xmax : float
        The upper bound for the domain that is to be collapsed

    Returns
    -------
    res : float
        The nanmean of the variance of the curves over the given interval
    """
    # Determine if all the data curves are the same length (have same #pts)
    if isinstance(xx, np.ndarray) and isinstance(yy, np.ndarray):
        same_length = True
    else:
        if all(len(i) == len(xx[0]) for i in xx) and all(len(i) == len(yy[0]) for i in yy):
            same_length = True
        else:
            same_length = False

    if check:
        print 'collapse_curves.py: same_length = ', same_length

    # Interpolate the curves
    xnew = np.linspace(xmin, xmax, nsamples)
    fv = np.zeros((len(xx), len(xnew)), dtype=float)
    ii = 0
    for xii in xx:
        # check that yy is a single valued function of xx by checking to see if there are duplicates in xx
        if len(xii) > len(np.unique(xii)):
            raise RuntimeError('Found duplicates in row ' + str(ii) + ' of xx. ' +
                               'yy should be a single valued function of xx for the residual ' +
                               'function to be helpful.')

        # For some reason interp1d doesn't let me use extrapolate for fill_values...
        # ff = interpolate.interp1d(xii, yy[ii], fill_value="extrapolate")
        # print 'xii = ', xii
        # print 'yy[ii] = ', yy[ii]
        ff = InterpolatedUnivariateSpline(xii, yy[ii], k=order)
        # plt.plot(xii, ff(xx[ii]), 'k.-')
        # plt.plot(xii, yy[ii], 'ro')
        # plt.show()
        # print 'ff = ', ff
        # print 'ff(', np.median(xnew), ')= ', ff(0.95)
        fv[ii] = ff(xnew)
        ii += 1

    # Compute residuals based on output
    res = np.nanmean(np.var(fv, axis=0))
    if check:
        for fvii in fv:
            plt.plot(xnew, fvii, '-')

        plt.title('average variance = ' + str(res))
        print 'fv = ', fv
        print 'np.shape(fv) = ', np.shape(fv)
        print 'var(fv, axis=0) = ', np.var(fv, axis=0)
        print 'var(fv, axis=1) = ', np.var(fv, axis=1)
        print 'np.sum(std(fv), axis=0) = ', np.sum(np.var(fv), axis=0)
        print 'np.sum(std(fv, axis=0)) = ', np.sum(np.var(fv, axis=0))
        plt.show()
        plt.clf()

    return res


def chisquared_curves(xx, yy, uncertainties, xmin, xmax, order=1, check=False, nsamples=100, eps=1e-14, replace_zeros=False):
    """Sum the difference from mean of interpolated curves (xx[ii, yy[ii]) over a sampling from xmin to xmax,
    for each curve (xx[ii], yy[ii]) in xx and yy. Divide by uncertainty at each point on curve.
    Note that yy[ii] must be a (single-valued) function of xx[ii] for this to be useful, for all ii.

    Parameters
    ----------
    xx : list of N float arrays (of possible variable length) or N x M float array
        The parameter that is varying across the phase boundary, spanning the domain of the data for each curve
    yy : list of N float arrays (of possible variable length) or N x M float array
        y coords for N curves with M points each (possibly variable #pts in each curve)
    xmin : float
        The lower bound for the domain that is to be collapsed
    xmax : float
        The upper bound for the domain that is to be collapsed

    Returns
    -------
    res : float
        The nanmean of the variance of the curves over the given interval
    """
    # Determine if all the data curves are the same length (have same #pts)
    if isinstance(xx, np.ndarray) and isinstance(yy, np.ndarray):
        same_length = True
    else:
        if all(len(i) == len(xx[0]) for i in xx) and all(len(i) == len(yy[0]) for i in yy):
            same_length = True
        else:
            same_length = False

    if check:
        print 'collapse_curves.py: same_length = ', same_length

    # Interpolate the curves
    xnew = np.linspace(xmin, xmax, nsamples)
    fv = np.zeros((len(xx), len(xnew)), dtype=float)
    ii = 0
    for xii in xx:
        # check that yy is a single valued function of xx by checking to see if there are duplicates in xx
        if len(xii) > len(np.unique(xii)):
            raise RuntimeError('Found duplicates in row ' + str(ii) + ' of xx. ' +
                               'yy should be a single valued function of xx for the residual ' +
                               'function to be helpful.')

        # For some reason interp1d doesn't let me use extrapolate for fill_values...
        # ff = interpolate.interp1d(xii, yy[ii], fill_value="extrapolate")
        # print 'xii = ', xii
        # print 'yy[ii] = ', yy[ii]
        ff = InterpolatedUnivariateSpline(xii, yy[ii], k=order)
        # plt.plot(xii, ff(xx[ii]), 'k.-')
        # plt.plot(xii, yy[ii], 'ro')
        # plt.show()
        # print 'ff = ', ff
        # print 'ff(', np.median(xnew), ')= ', ff(0.95)
        fv[ii] = ff(xnew)
        ii += 1

    # Compute residuals based on output
    meancurv = np.mean(fv, axis=0)
    meaninterp = InterpolatedUnivariateSpline(xnew, meancurv, k=order)

    # Now for each observation point, query the distance from the mean curv
    num = 0
    chisq = 0
    for (xii, yii, uii) in zip(xx, yy, uncertainties):
        if replace_zeros:
            inds = np.where(np.logical_and(xii > xmin, xii < xmax))[0]
            uii[np.where(uii < eps)[0]] = np.min(uii[uii > eps])
        else:
            inds = np.where(np.logical_and(np.logical_and(xii > xmin, xii < xmax), uii > eps))[0]
        # print(inds)
        # print(np.shape(xii[inds]))
        print('uncertainty = ' + str(uii[inds]**2))
        chisq += np.sum((yii[inds] - meaninterp(xii[inds])) ** 2 / (uii[inds]**2))
        print(chisq)
        num += np.size(inds)

    if check:
        # print(fv)
        plt.plot(xnew, meancurv, 'k.')
        for (fvii, ii) in zip(fv, range(len(fv))):
            plt.plot(xnew, fvii, '--')
            plt.plot(xx[ii], yy[ii], 'o')

        # print 'fv = ', fv
        # print 'np.shape(fv) = ', np.shape(fv)
        plt.title('num=' + str(num) + ' chisq=' + str(chisq))
        plt.pause(0.00001)
        plt.clf()

    return chisq, num


###################################################################################################
# Get error from the curvature of the standard (quadrature) errors wrt the best fit, varying nu
###################################################################################################
def nu_error_asymptotic_parabolas(xx, yy,  sz, sz0, xcstar, nustar, xmin, xmax, mindnu_frac, maxdnu_frac, nevals,
                                  var_min, check=False, view=False, pausetime=0.2):
    """Fit the radius of curvature at xc of the curve yy(xx). Do this nevals times for different domain widths (from
    minrange to maxrange) of xx near xc with initial guesses for fit params p0=(p[0], p[1]).
    Note that in the code for this function unc refers to uncertainty from fitting residuals to a quadratic

    Parameters
    ----------

    """
    if check or view:
        plt.close('all')
        nuv = np.linspace(nustar * (1. - maxdnu_frac * 0.5), nustar * (1. + maxdnu_frac * 0.5), 50)
        resv = []
        for nu in nuv:
            xrescale = finsz_fixed_xc(nu, xx, sz, sz0, xcstar)
            var = variance_curves(xrescale, yy, xmin, xmax, order=1, check=False)
            resv.append(var)

        resv = np.array(resv)
        plt.plot(nuv, resv, '.-')
        plt.ylabel(r'Variance in curves, $\sigma^2$')
        plt.xlabel(r'Exponent $\nu$')
        plt.title(r'Estimating error from plot: $\nu^* = $' + str(nustar))

    xrv = np.linspace(mindnu_frac, maxdnu_frac, nevals)
    nu_uncs = np.zeros(len(xrv), dtype=float)
    errs = np.zeros(len(xrv), dtype=float)
    ii = 0
    for xr in xrv:
        nulo = nustar * (1. - xr * 0.5)
        nuhi = nustar * (1. + xr * 0.5)

        # Error of nearby values of nu, for the original data xx, sz, etc
        nuv = np.linspace(nulo, nuhi, 50)
        resv = []
        for nu in nuv:
            xrescale = finsz_fixed_xc(nu, xx, sz, sz0, xcstar)
            var = variance_curves(xrescale, yy, xmin, xmax, order=1, check=False)
            resv.append(var)

        resv = np.array(resv)
        unc, err = err_from_quadfit(nuv, resv, nustar, var_min, check=False)
        nu_uncs[ii] = unc
        errs[ii] = err
        ii += 1

    if check or view:
        if check:
            plt.show()
        else:
            plt.pause(pausetime)
        # Get weighted sum of uncertainty in nu
        nu_unc = np.sum(nu_uncs / errs) / np.sum(1. / errs)
        plt.plot(xrv, nu_uncs, 'k.-')
        plt.xlabel(r'fractional range of $\nu^*$')
        plt.ylabel(r'Residuals from guesses near $\nu^*$')
        plt.title(r'Showing Residuals vs range of $\nu$. Found $\sigma_\nu=$' + '{0:0.3f}'.format(nu_unc))
        if check:
            plt.show()
        else:
            plt.pause(pausetime)

    return nu_uncs, errs


def xc_error_asymptotic_parabolas(xx, yy,  sz, sz0, xcstar, nustar, xmin, xmax, mindxc_frac, maxdxc_frac, nevals,
                                  var_min, check=False, view=False, pausetime=0.2):
    """Fit the radius of curvature at xc of the curve yy(xx). Do this nevals times for different domain widths (from
    minrange to maxrange) of xx near xc with initial guesses for fit params p0=(p[0], p[1]).

    Parameters
    ----------

    Returns
    -------
    xc_uncs :
    errs :
    """
    if check or view:
        plt.close('all')
        xcv = np.linspace(xcstar * (1. - maxdxc_frac * 0.5), xcstar * (1. + maxdxc_frac * 0.5), 50)
        resv = []
        for xc in xcv:
            xrescale = finsz_fixed_xc(nustar, xx, sz, sz0, xc)
            var = variance_curves(xrescale, yy, xmin, xmax, order=1, check=False)
            resv.append(var)

        resv = np.array(resv)
        plt.plot(xcv, resv, '.-')
        plt.ylabel(r'Variance in curves, $\sigma^2$')
        plt.xlabel(r'Transition location $x_c$')
        plt.title(r'Estimating error from plot: $x_c^* = $' + str(xcstar))

    xrv = np.linspace(mindxc_frac, maxdxc_frac, nevals)
    xc_uncs = np.zeros(len(xrv), dtype=float)
    errs = np.zeros(len(xrv), dtype=float)
    ii = 0
    for xr in xrv:
        xclo = xcstar * (1. - xr * 0.5)
        xchi = xcstar * (1. + xr * 0.5)

        # Error of nearby values of nu, for the original data xx, sz, etc
        xcv = np.linspace(xclo, xchi, 50)
        resv = []
        for xc in xcv:
            xrescale = finsz_fixed_xc(nustar, xx, sz, sz0, xc)
            var = variance_curves(xrescale, yy, xmin, xmax, order=1, check=False)
            resv.append(var)

        resv = np.array(resv)
        unc, err = err_from_quadfit(xcv, resv, xcstar, var_min, check=False)
        xc_uncs[ii] = unc
        errs[ii] = err
        ii += 1

    if check or view:
        if check:
            plt.show()
        else:
            plt.pause(pausetime)
        xc_unc = np.sum(xc_uncs / errs) / np.sum(1. / errs)
        plt.plot(xrv, xc_uncs, 'k.-')
        plt.plot(xrv, xc_uncs + np.sqrt(errs), 'r.-')
        plt.plot(xrv, xc_uncs - np.sqrt(errs), 'r.-')
        plt.xlabel(r'fractional range of $x_c^*$')
        plt.ylabel(r'Residuals from guesses near $x_c^*$')
        plt.title(r'Showing Residuals vs range of $x_c$. Found $\sigma_{x_c}=$' + '{0:0.3f}'.format(xc_unc))
        if check:
            plt.show()
        else:
            plt.pause(pausetime)

    return xc_uncs, errs


def err_from_quadfit(xv, resv, xstar, residuals_min, check=False, show=False):
    """For finite size scaling, measure the uncertainty near the minimum of the Chi^2 function, and report the
    uncertainty in the fit to the quadratic minium.

    Parameters
    ----------
    xv : n x 1 float array
        values of the fit parameter
    resv : n x 1 float array
        the residuals vector, chi^2 / N for each value of fit parameter xv.
        If multivariate function, one should fit each residual vs fit param to a parabola
    xstar : float
        the critical (hopefully minimum) value of xv at which the parabola is measured
    residuals_min : float
        the minimum y value (min residual value)
    check : bool
        show intermediate results
    show : bool
        Show a plot of the parabolic fit
    """
    # Find second derivative evaluated at nu_min
    quadfun = make_parabola_xmin(xstar)
    pout, cov = optimize.curve_fit(quadfun, xv, resv, p0=[1., residuals_min])
    err = np.sum((resv - quadfun(xv, pout[0], pout[1]))**2)
    # pout, cov = optimize.curve_fit(make_parabola_xymin(nustar, var_min), nuv, resv, p0=[1.])
    # print 'pout = ', pout
    unc = 2. / pout[0]

    if check:
        print 'pout = ', pout
        print 'cov = ', cov
        plt.plot(nuv, pout[0] * (nuv - nustar) ** 2 + pout[1], '-')
        # plt.plot(nuv, pout[0] * (nuv - nustar) ** 2 + var_min, '-')
        plt.pause(0.1)
        if show:
            plt.show()

    return unc, err


def make_parabola_xmin(xmin):
    """Create and return a function which is a parabola centered around a known x coordinate, xmin"""
    def parabola_xmin(xx, p0, p1):
        return p0 * (xx - xmin)**2 + p1
    return parabola_xmin


def make_parabola_xymin(xmin, ymin):
    """Create and return a function which is a parabola centered around a known xy coordinate, (xmin, ymin)"""
    def parabola_xmin(xx, p0):
        return p0 * (xx - xmin)**2 + ymin
    return parabola_xmin


######################################################
# Functions for plotting the results of the collapse
######################################################
def plot_curve_collapse(xx, yy, sz, sz0, nu, xc, ax=None, title=None, outname=None, dpi=300, show=False, pause=False,
                        alpha=None, **kwargs):
    """Single collapse optimization: plot the collapsed curves with curves colored by finite size given the value of
    rescaling exponent, nu, and the critical value of x at which the phase changes, xc.

    Parameters
    ----------
    xx : list of N float arrays (of possible variable length) or N x M float array
        The parameter that is varying across the phase boundary, spanning the domain of the data for each curve
    yy : list of N float arrays (of possible variable length) or N x M float array
        y coords for N curves with M points each (possibly variable #pts in each curve)
    sz : list
        The system sizes (one value for each curve)
    sz0 : float
        The reference system size
    nu : float
        The finite size scaling exponent
    xc : float
        The critical value of the parameter varying across the phase boundary
    outname : str, list of strings, or None
        The output path to save a rendering of the figure. If a list, figure is saved as each string path
    show : bool
        If True, show the plot (forcing user to close it before continuing)
    **kwargs : keyword arguments for lepm.plotting.plotting.initialize_1panel_fig()

    Returns
    -------
    """
    if ax is None:
        fig, ax = leplt.initialize_1panel_fig(**kwargs)
    if alpha is None:
        alpha = 1. / len(xx)

    # Inspect validity/collapse of finite size scaling results
    cmap = plt.get_cmap('copper')
    colors = cmap(np.linspace(0., 1., len(sz)))
    xrescale = finsz_fixed_xc(nu, xx, sz, sz0, xc)
    for ii in range(len(xx)):
        plt.plot(xrescale[ii], yy[ii], '.', color=colors[ii], alpha=alpha,
                 label=r'$L = $' + '{0:0.2f}'.format(sz[ii]))

    # Make title and legend
    if title is None:
        # plt.suptitle(r'Finite-size scaling collapse: $\nu = $' + str(nu))
        ax.text(0.5, 1.05, r'Finite-size scaling collapse: $\nu = $' + str(nu), transform=ax.transAxes)
    else:
        # plt.suptitle(title)
        ax.text(0.5, 1.05, title, transform=ax.transAxes)

    ax.legend(loc='center left', bbox_to_anchor=(1., .5))

    if outname is not None and outname is not 'none' and outname is not '':
        if isinstance(outname, list):
            for out in outname:
                plt.savefig(out, dpi=dpi)
        else:
            plt.savefig(outname, dpi=dpi)
    if show:
        plt.show()
    if pause:
        plt.pause(0.1)
    return ax


def plot_param_vs_range(range_arr, nu, nu_err, xc, xc_err=None, ax=None, figlabs=None, outname=None, dpi=300,
                        show=False, **kwargs):
    """Plot the parameters which collapse the curves as a function of range size

    Parameters
    ----------
    nulist : list
    xclist : list
    outname : str, list of strings, or None
        The output path to save a rendering of the figure. If a list, figure is saved as each string path
    show : bool
        If True, show the plot (forcing user to close it before continuing)
    **kwargs : keyword arguments for lepm.plotting.plotting.initialize_1panel_fig()

    Returns
    -------
    """
    # check if xc is a constant across all collapses
    xcconstv = np.array([np.array(xcii == xc[0]).all() for xcii in xc])
    xcconst = xcconstv.all()

    # unpack figure labels
    if figlabs is None:
        figlabs = {'xlabel': r'Range of collapse, $\xi$',
                   'title': r'Finite-size scaling dependence on range, $\xi$'}
        if xcconst:
            figlabs['ylabel'] = r'Finitize-size exponent, $\nu$'
        else:
            figlabs['ylabel'] = r'Scaling exponent, $\nu$, and critical value, $x_c$'
    else:
        if 'xlabel' not in figlabs:
            figlabs['xlabel'] = None
        if 'ylabel' not in figlabs:
            if xcconst:
                figlabs['ylabel'] = r'Finitize-size exponent, $\nu$'
            else:
                figlabs['ylabel'] = r'Scaling exponent, $\nu$, and critical value, $x_c$'
        if 'title' not in figlabs:
            figlabs['title'] = r'Finite-size scaling dependence on range, $\xi$'

    if ax is None:
        fig, ax = leplt.initialize_1panel_fig(**kwargs)

    # First make a plot of all the collapsed curves, colored by range
    print 'nulist = ', nu
    print 'nuerr = ', nu_err

    # If xc was varied across the different collapses, plot how it is changing with range size
    if xcconst:
        ax.errorbar(range_arr, nu, yerr=nu_err, fmt='.', ecolor='g', label=r'$\nu$', capsize=2)
        ylims = (0, np.max(nu) * 1.1)
    else:
        ax.errorbar(range_arr, nu, yerr=nu_err, fmt='b.', ecolor='b', label=r'$\nu$', capsize=2)
        ax.set_ylim(0, max(np.max(nu), np.max(xc)) * 1.1)
        ax.errorbar(range_arr, xc, yerr=xc_err, fmt='g.', ecolor='g', label=r'$x_c$', capsize=2)
        ax.legend(loc='best', fancybox=True)
        # Set ylims for zoom in
        ylims = (0, max(np.max(xc), np.max(nu)))

    # Axis labels
    ax.set_xlabel(figlabs['xlabel'])
    ax.set_ylabel(figlabs['ylabel'])

    # Make title and legend
    plt.suptitle(figlabs['title'])
    # ax.legend(loc='center left', bbox_to_anchor=(1., .5))

    if outname is not None and outname is not 'none' and outname is not '':
        if isinstance(outname, list):
            for out in outname:
                plt.savefig(out, dpi=dpi)
            ax.set_ylim(ylims)
            for out in outname:
                # insert '_zoom' between file name and png
                plt.savefig(out.split('.p')[0] + '_zoom.p' + out.split('.p')[1], dpi=dpi)
        else:
            plt.savefig(outname, dpi=dpi)
            ax.set_ylim(ylims)
            # insert '_zoom' between file name and png
            plt.savefig(out.split('.p')[0] + '_zoom.p' + out.split('.p')[1], dpi=dpi)
    if show:
        plt.show()
    return ax


def  plot_curve_collapse_sequence(range_arr, xx, yy, sz, sz0, nulist, xclist, ax=None, figlabs=None, outname=None,
                                 dpi=300, show=False, **kwargs):
    """Plot collections of  collapsed curves with curves colored by the width of the optimization window, xi, given
    the value of rescaling exponent, nu, and the critical value of x at which the phase changes, xc.

    Parameters
    ----------
    xx : list of N float arrays (of possible variable length) or N x M float array
        The parameter that is varying across the phase boundary, spanning the domain of the data for each curve
    yy : list of N float arrays (of possible variable length) or N x M float array
        y coords for N curves with M points each (possibly variable #pts in each curve)
    sz : list
        The system sizes (one value for each curve)
    sz0 : float
        The reference system size
    nulist : list
    xclist : list
    ax : matplotlib.axis instance
    figlabs : dict
    outname : str, list of strings, or None
        The output path to save a rendering of the figure. If a list, figure is saved as each string path
    show : bool
        If True, show the plot (forcing user to close it before continuing)
    **kwargs : keyword arguments for lepm.plotting.plotting.initialize_1panel_fig()

    Returns
    -------
    ax : the matplotlib axis instance
    """
    # unpack figure labels
    if figlabs is None:
        figlabs = {'xlabel': r'Rescaled parameter, $x_c + (x - x_c)\left(\frac{L}{L_0}\right)^{1/\nu}$',
                   'ylabel': r'Chern measurement, $\langle C \rangle$',
                   'title': r'Finite-size scaling dependence on range, $\xi$'}
    else:
        if 'xlabel' not in figlabs:
            figlabs['xlabel'] = None
        if 'ylabel' not in figlabs:
            figlabs['ylabel'] = r'Chern measurement, $C$'
        if 'title' not in figlabs:
            figlabs['title'] = r'Finite-size scaling dependence on range, $\xi$'

    if ax is None:
        fig, ax = leplt.initialize_1panel_fig(**kwargs)

    # First make a plot of all the collapsed curves, colored by range
    cmap = plt.get_cmap('copper')
    colors = cmap(np.linspace(0., 1., len(nulist)))
    kk = 0
    markerlist = leplt.get_markerstyles(len(nulist))
    alpha = float(1./float(len(nulist)))
    for nu in nulist:
        xrescale = finsz_fixed_xc(nu, xx, sz, sz0, xclist[kk])
        print 'collapse_curves: kk = ', kk, ': range_arr[kk] = ', range_arr[kk]
        for ii in range(len(xx)):
            if ii == 0:
                ax.plot(xrescale[ii], yy[ii], linestyle='-', marker=markerlist[kk], color=colors[kk], alpha=alpha,
                        label=r'$\xi = $' + '{0:0.2f}'.format(range_arr[kk]))
            else:
                ax.plot(xrescale[ii], yy[ii], linestyle='-', marker=markerlist[kk], color=colors[kk], alpha=alpha)
        kk += 1

    # Axis labels
    ax.set_xlabel(figlabs['xlabel'])
    ax.set_ylabel(figlabs['ylabel'])

    # Make title and legend
    ax.text(0.5, 1.05, figlabs['title'], transform=ax.transAxes)
    ax.legend(loc='center left', bbox_to_anchor=(1., .5))

    if outname is not None and outname is not 'none' and outname is not '':
        if isinstance(outname, list):
            for out in outname:
                plt.savefig(out, dpi=dpi)
        else:
            plt.savefig(outname, dpi=dpi)
    if show:
        plt.show()
    return ax


if __name__ == "__main__":
    #########################################################################
    # Check variance of curves
    #########################################################################
    xx, yy = [], []
    for lng in [3, 6, 9]:
        xx.append(np.linspace(0, 1, lng))
        yy.append(np.linspace(0, lng, lng))

    out = variance_curves(xx, yy, xmin=0.0, xmax=0.2)
    print 'out = ', out
    out = variance_curves(xx, yy, xmin=0.2, xmax=0.4)
    print 'out = ', out
    out = variance_curves(xx, yy, xmin=0.4, xmax=0.6)
    print 'out = ', out
    out = variance_curves(xx, yy, xmin=0.6, xmax=0.8)
    print 'out = ', out
    out = variance_curves(xx, yy, xmin=0.8, xmax=1.0)
    print 'out = ', out
    out = variance_curves(xx, yy, xmin=0., xmax=1.)
    print 'out = ', out
    sys.exit()

    xx, yy = [], []
    for lng in [3, 6, 9]:
        xx.append(np.linspace(0, 1, lng))
        yy.append(lng * np.ones(lng))

    out = variance_curves(xx, yy, xmin=0, xmax=1., check=True)
    print 'out = ', out

    #########################################################################
    # Collapsing 3 curves with same # data pts in each -- Fermi Functions
    #########################################################################
    # Do one example of finite size scaling in which all curves (arrays) are the same length (same number of points)
    x0 = np.arange(0, 5., 0.10)
    x1 = np.arange(0, 5.5, 0.11)
    x2 = np.arange(0, 6.5, 0.13)
    y0 = fermi_function(x0, 2.5, .3)
    y1 = fermi_function(x1, 2.5, .2)
    y2 = fermi_function(x2, 2.5, .1)
    xx = np.dstack((x0, x1, x2))[0].T
    yy = np.dstack((y0, y1, y2))[0].T

    # Initial guess and params/bounds for fit
    p0 = [2.5]
    xmin = 1.5
    xmax = 3.5
    xc = 2.5
    sz = [10., 30., 50.]
    sz0 = 30.

    # Collapse the curves
    xrescale = finsz_fixed_xc(p0[0], xx, sz, sz0, xc)
    # p0, xx, yy, sz, sz0, deltax, apply_bounds = True, order = 1, check = False, view = False
    nu, unc = minimize_variance_curves_fixed_xc(p0, xx, yy, sz, sz0, xmin, xmax, xc, apply_bounds=True, check=True)

    ################################################################
    # Collapsing 2 curves of different length -- Fermi Functions
    ################################################################
    # Do one example of finite size scaling in which all curves (arrays) are the same length (same number of points)
    x0 = np.arange(0, 5., 0.10)
    x1 = np.arange(0, 5., 0.12)
    xx = [x0, x1]
    yy = [fermi_function(x0, 2.5, .3), fermi_function(x1, 2.5, .2)]

    # Initial guess and params/bounds for fit
    p0 = [2.4]
    xmin = 2.
    xmax = 3.
    xc = 2.5
    sz = [10., 50.]
    sz0 = 30.

    # Collapse the curves
    nu, unc = minimize_variance_curves_fixed_xc(p0, xx, yy, sz, sz0, xmin, xmax, xc, apply_bounds=True, check=True)
