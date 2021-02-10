import numpy as np
import shapely.geometry as sg
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from scipy import fftpack
import scipy
##################
import lepm.lattice_elasticity as le
import lepm.dataio as dio
import lepm.plotting.plotting as leplt
import lepm.plotting.colormaps as cmaps
import descartes
import subprocess

'''
Functions for calculating the structure of a lattice:
    -- density (number) variance of particles
    -- structure factor
    -- two-pt correlation functions
'''


def bond_length_histogram(xy, NL, KL, BL, PVx=[], PVy=[], fig=None, ax=None, outdir=None, check=False, savetxt=True):
    """Histogram the bond lengths of the lattice.
    If fig or axis is not None, adds to that fig/axis.
    
    Parameters
    ----------
    xy : NP x 2 float array
        positions of point set
    NL : NP x max(#neighbors) int array
        The ith row contains indices for the neighbors for the ith point
    KL : NP x max(#neighbors) int array
        spring connection/constant list, where 1 corresponds to a true connection,
        0 signifies that there is not a connection, -1 signifies periodic bond
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points. Negative values denote particles connected
        through periodic bonds.
    PVx : NP x NN float array (optional, for periodic lattices)
        ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen by particle i
    PVy : NP x NN float array (optional, for periodic lattices)
        ijth element of PVy is the y-component of the vector taking NL[i,j] to its image as seen by particle i
    fig : matplotlib figure instance or None
        The figure in which to plot the bond length histogram. If None, uses current figure
    ax : matplotlib axis instance or None
        The axis in which to plot the histogram. If None, uses current axis.
    outdir : string or None
        The file path in which to save the figure, if not None
    check : bool
        Whether to view intermediate results
    savetxt: bool
        Whether to save the bin values and counts as a text file. Outdir must be specified for this to have an effect
        when True.

    Returns
    ----------
    bL : Nbonds x 1 float array
        bond lengths of the lattice
    """
    if fig is None and ax is None:
        close_fig = True
        fig = plt.gcf()
        fig.clf()
        ax = plt.gca()
    elif fig is None and ax is not None:
        close_fig = False
        fig = plt.gcf()
    else:
        close_fig = False

    if outdir is not None:
        infodir = outdir
        saveout = True
        le.ensure_dir(infodir)
    else:
        saveout = False
    
    # Make dir for lattice info
    BM = le.NL2BM(xy, NL,KL, PVx=PVx, PVy=PVy)
    bL = le.BM2bL(NL,BM,BL)
    nbonds, bins, patches = ax.hist(bL, bins=int(len(bL)*0.5) )
    #ax.set_title('Bond Lengths')
    ax.set_xlabel('Bond Length')
    ax.set_ylabel('Frequency')
    
    if saveout:
        plt.savefig(infodir+'bond_lengths.png')
    if check:
        plt.show()
    if close_fig:
        plt.clf()

    if savetxt and saveout:
        print 'Calling bond_length_histogram with savetxt == True'
        print 'savetxt =', savetxt
        # get centers of each bin
        binc = 0.5 * ( bins[1:] + bins[:-1] )
        MM = np.dstack((binc, nbonds))[0]
        header = 'Bond length histogram: center of bin, number of bonds in that bin'
        np.savetxt(infodir+'bond_length_hist.txt',MM,fmt='%.18e %i', header=header, delimiter=',')

    return bL, nbonds, bins
    

def calc_number_variance(xy, LL, outdir=None, fit=True, check=False):
    """
    Note: this only works given a rectangular LLv (square system size).
    Also assumes lattice is centered about the origin (0,0)
    
    Parameters
    ----------
    xy : NP x 2 array
        positions of point set
    LL : tuple
        The system size (bounding box dimensions), assuming system is a square (Xsize, Ysize)
    outdir : string or None
        if not None, path to where plots will be saved
    
    Returns
    ----------
    radV : N x 1 float array
        radius vector, the radii used in circles that sample the points in the lattice
    varN : N x 1 float array
        variance in number of particles within regions of size radV
    a_regr : float
        scaling exponent in power law varN ~ radV^{a_regr}
    """
    if outdir is not None:
        infodir = outdir
        saveout = True
    else:
        saveout = False
    
    LLmin = np.min(np.abs(LL))
    # Define an array of radii
    radV = np.arange(6, LL[0] * 0.17, 0.5)
    print 'radV = ', radV
    varN = np.zeros_like(radV)
    for ii in range(len(radV)):
        # Create a grid of circles for each radV value
        rad = radV[ii]
        # Be conservative with the number of circles sampling the space to avoid the edges
        nrows = np.floor(LLmin/(2.*rad))-1 
        print 'computing variance for radius=', rad
        # print 'LLmin = ', LLmin
        # print 'rad = ', rad
        # print 'nrows = ', nrows
        centers = np.arange(-nrows*rad,nrows*rad,2.*rad)
        xcir, ycir = np.meshgrid(centers, centers)
        xycir = np.dstack( (xcir.ravel(), ycir.ravel() ))[0]
        xycir -= np.mean(xycir, axis=0) + np.random.rand(len(xycir),2)
        # Compute number of points in each circle
        N = np.array([ len(np.where( (xy[:,0]-xycir[jj,0])**2 + (xy[:,1]-xycir[jj,1])**2 < rad**2 )[0]) for jj in range(len(xycir)) ])
        varN[ii] = (np.mean(N**2) - np.mean(N)**2) / np.mean(N)
        if check:
            ax = plt.gca()
            patches = []
            for jj in range(len(xycir)):
                circle = mpatches.Circle((xycir[jj,0], xycir[jj,1]), rad)
                patches.append(circle)
            
            colors = 100*np.random.rand(len(patches))
            p = PatchCollection(patches, alpha=0.4)
            p.set_array(np.array(colors))
            ax.add_collection(p)
            ax.scatter(xy[:, 0], xy[:, 1], s=10, edgecolor='None', facecolor='b')
            print 'N =', N
            print 'varN =', varN 
            plt.show()
        
    logvarN = np.log(varN)
    logradV = np.log(radV)
    
    if saveout:
        # Plot variance of N
        plt.clf()
        plt.plot(radV, varN, 'r.-')
        plt.title(r'$\sigma^2(R) = \frac{< N^2 > - < N >^2}{< N > }$')
        plt.xlabel(r'$R$')
        plt.ylabel(r'$\sigma^2$')
        plt.ylim(0.0, np.max(varN)*1.2)
        plt.savefig(infodir + 'varianceN.png')
        if check:
            plt.show()
        plt.clf()
        
        # Plot variance of N log-linear
        plt.clf()
        plt.plot(logradV, logvarN, 'r.-')
        plt.title(r'$\sigma^2(R) = \frac{< N^2 > - < N >^2}{< N > }$')
        plt.xlabel(r'$\ln R$')
        plt.ylabel(r'$\ln \sigma^2$')
        # plt.ylim(0.0, np.max(varN)*1.2)
        plt.savefig(infodir + 'varianceN_loglog.png')
        plt.clf()
        
        # Save variance as txt file
        MM = np.dstack((radV,varN))[0]
        np.savetxt(infodir+'varN.txt', MM, delimiter=',', header='radius, (<N**2>-<N>**2)/<N> : particle variance for point set')
    
    # Fit to linear regression
    if fit and np.isfinite(logvarN).size > 0:
        (a_regr, b_regr) = scipy.polyfit(logradV[ np.isfinite(logvarN) ], logvarN[ np.isfinite(logvarN) ], 1)
        logsig_regr = scipy.polyval([a_regr, b_regr],logradV)
        
        if saveout:
            plt.plot(logradV, logsig_regr, 'k-')        
            textstr = r'$\ln \sigma^2 \sim$ '+'{0:0.3f}'.format(a_regr) + r' $\ln R$'
            print 'textstr = ', textstr
            xstr, ystr = np.median(logradV), max( np.max(logsig_regr), np.max(logvarN) )
            print 'xstr =', xstr, '  ystr = ', ystr
            plt.text(xstr, ystr, textstr)
            plt.savefig(infodir + 'varianceN_loglogfit.png')
            if check:
                plt.show()
            plt.clf()
    else:
        a_regr = 0
    
    return radV, varN, a_regr


def calc_gxy_gr(xy, BBox, outdir=None, dr=0.1, eps=1e-9, maxgxy=0.04, distlimit=20.0, check=False):
    """
    Calculate two-point correlation function in x,y and r using a small portion of the sample (~1/9th)
    Assumes the lattice is centered about the origin.

    Parameters
    ----------

    Returns
    ----------
    """
    # First decide whether we are saving output or not. If so, register colormap
    if outdir is not None:
        infodir = outdir
        saveout = True
        # Register cmaps if not registered, and set colormap
        if 'viridis' not in plt.colormaps():
            plt.register_cmap(name='viridis', cmap=cmaps.viridis)
        
        plt.set_cmap(cmaps.viridis)
    else:
        saveout = False

    NP = len(xy) 
    gxyL = .5 * np.min(np.abs(BBox))
    gxylim = min(gxyL, distlimit)
    xvals = np.arange(-gxylim, gxylim+dr*0.5, dr)
    yvals = np.arange(-gxylim, gxylim+dr*0.5, dr)
    
    BBox_sg = sg.Polygon([tuple(row) for row in BBox])
    Area = BBox_sg.area
    density = float(NP)/Area
    print 'density = ', density
    
    # Consider only particles inside box which is a distance gxyL from the closest edge of BBox.
    # If there are NIP of these particles, then we will have an NIP x NP x 2 array for x,y distances to other particles
    # which we then will histogram. Easiest to pick gxyL = .25*LL[0] as done above if sample is square.
    # However, we can do better if the system is large!
    # If gxylim < gxyL, let's use all particles except those < gxyL away from the edge, and take gxyL to be
    # gxyL = gxylim = min(0.5*min(np.abs(BBox), 20.0). Now we have an (NH - 2*gxyL) width box of particles to use.
    # The way we can get away with using such a large box is to only look at particles (j) less than gxyL away from
    # each particle (i).

    if gxylim < gxyL:
        gxyL = gxylim
        # First define plentyN, which should be more than enough cols for each row (> # distances measured from each
        # particle). Trailing zeros in measured distances will be eliminated later.
        furthest = np.min(np.abs(BBox))-gxyL
        ipIND = np.where(np.logical_and(np.abs(xy[:, 0]) < furthest, np.abs(xy[:, 1]) < furthest))[0]
        len_xyip = len(xy[ipIND, 0])

        # Define regions which will be considered one by one: top left, then top left + one box to right, ...
        # number of boxes in horizonal and vertical dims
        nbH = int(np.ceil((np.max(xy[ipIND, 0]) - np.min(xy[ipIND, 0])) / gxyL))
        nbV = int(np.ceil((np.max(xy[ipIND, 1]) - np.min(xy[ipIND, 1])) / gxyL))
        edgesX = [min(-furthest + gxyL * ii, furthest) for ii in range(nbH + 1)]
        edgesY = [min(-furthest + gxyL * ii, furthest) for ii in range(nbV + 1)]
        gxy_xVij = [[]] * (nbH * nbV)
        gxy_yVij = [[]] * (nbH * nbV)
        for ii in range(nbH):
            print 'Computing g(x,y) for row ', ii, 'of ', nbH
            for jj in range(nbV):
                inX = np.logical_and(xy[:, 0] > edgesX[ii], xy[:, 0] < edgesX[ii+1])
                inY = np.logical_and(xy[:, 1] > edgesY[jj], xy[:, 1] < edgesY[jj+1])
                reg = np.where(np.logical_and(inX, inY))[0]
                inX2 = np.logical_and(xy[:, 0] > (edgesX[ii] - gxyL), xy[:, 0] < edgesX[ii+1] + gxyL)
                inY2 = np.logical_and(xy[:, 1] > (edgesY[jj] - gxyL), xy[:, 1] < edgesY[jj+1] + gxyL)
                nbrs = np.where(np.logical_and(inX2, inY2))[0]

                # Each row is x distance from ith particle
                gxy_Xarr = np.ones((len(xy[reg]), len(xy[nbrs])), dtype=float)*xy[nbrs,0]
                gxy_Yarr = np.ones((len(xy[reg]), len(xy[nbrs])), dtype=float)*xy[nbrs,1]
                # print 'Computing g(x)...'
                # gxy_x = np.array([gxy_Xarr[i] - xyip[i,0] for i in range(len(xyip)) ])
                gxy_x = gxy_Xarr - np.dstack(np.array([xy[reg, 0].tolist()]*np.shape(gxy_Xarr)[1]))[0]
                # print 'Computing g(y)...'
                # gxy_y = np.array([gxy_Yarr[i] - xyip[i,1] for i in range(len(xyip)) ])
                gxy_y = gxy_Yarr - np.dstack(np.array([xy[reg, 1].tolist()]*np.shape(gxy_Xarr)[1]))[0]
                gxy_xVij[ii*nbV + jj] = gxy_x.ravel()
                gxy_yVij[ii*nbV + jj] = gxy_y.ravel()

        gxy_xV = np.hstack(tuple(gxy_xVij))
        gxy_yV = np.hstack(tuple(gxy_yVij))
        if check:
            print 'gxy_xV = ', gxy_xV
            print 'gxy_yV = ', gxy_yV

    else:
        # The system is small enough to handle all points simultaneously.
        ipIND = np.where(np.logical_and(np.abs(xy[:, 0]) < gxyL, np.abs(xy[:, 1]) < gxyL))[0]
        xyip = xy[ipIND, :]
        # Each row is x distance from ith particle
        gxy_Xarr = np.ones((len(xyip), len(xy)), dtype=float)*xy[:, 0]
        gxy_Yarr = np.ones((len(xyip), len(xy)), dtype=float)*xy[:, 1]
        print 'Computing g(x)...'
        # gxy_x = np.array([gxy_Xarr[i] - xyip[i,0] for i in range(len(xyip)) ])
        gxy_x = gxy_Xarr - np.dstack(np.array([xyip[:, 0].tolist()]*np.shape(gxy_Xarr)[1]))[0]
        print 'Computing g(y)...'
        # gxy_y = np.array([gxy_Yarr[i] - xyip[i,1] for i in range(len(xyip)) ])
        gxy_y = gxy_Yarr - np.dstack(np.array([xyip[:, 1].tolist()]*np.shape(gxy_Xarr)[1]))[0]
        gxy_xV = gxy_x.ravel()
        gxy_yV = gxy_y.ravel()
        len_xyip = len(xyip)

    # Remove delta function from distance to self
    print 'Removing delta function at g(0)...'
    nonorigin = np.where(np.logical_and(np.abs(gxy_xV) > eps, np.abs(gxy_yV) > eps))[0]
    # if check:
    #     originpts = np.where(np.logical_and( np.abs(gxy_xV) < eps, np.abs(gxy_yV) < eps))[0]
    #     print 'originpts = ', originpts
    #     print 'len(originpts) = ', len(originpts)
    #     print 'len(xyip) = ', len(xyip)
    #     gxyV = np.dstack((gxy_xV,gxy_yV))[0]
    #     print 'gxyV[originpts] =', gxyV[originpts]

    # Creating a new object to remove the delta function is unnecessary and memory-intensive
    # gxy_xV = gxy_xV[nonorigin]
    # gxy_yV = gxy_yV[nonorigin]

    print 'Constructing histogram g(x,y)...'
    gxy_grid, xedges, yedges = np.histogram2d(gxy_xV[nonorigin], gxy_yV[nonorigin], bins=(xvals, yvals))
    xcenters = xedges[:-1] + 0.5 * (xedges[1:] - xedges[:-1])
    ycenters = yedges[:-1] + 0.5 * (yedges[1:] - yedges[:-1])
    gxy_grid = gxy_grid.astype(float)
    # I don't think I should divide by density here:
    gxy_grid /= float(len_xyip)
    gxy = gxy_grid.ravel()
    xgrid, ygrid = np.meshgrid(xedges, yedges)
    xcgrid, ycgrid = np.meshgrid(xcenters, ycenters)
    xc, yc = xcgrid.ravel(), ycgrid.ravel()
        
    # Subtract off delta function at the center
    # origin = np.where(np.logical_and( np.abs(xvals)==np.min(np.abs(xvals)), np.abs(yvals)==np.min(np.abs(yvals)) ))[0]
    # print '\n\norigin = ', origin, '\n\n'
    # gxy[origin] = 0.
    # print 'shape(xedges) = ', np.shape(xedges), ' -> ', xedges
    # print 'shape(yedges) = ', np.shape(yedges), ' -> ', yedges
    # print 'shape(yvals) = ', np.shape(yvals), ' -> ', yvals
    # print 'shape(xvals) = ', np.shape(xvals), ' -> ', xvals
    # print 'shape(xgrid) = ', np.shape(xgrid), ' -> ', xgrid
    # print 'shape(ygrid) = ', np.shape(ygrid), ' -> ', ygrid
    # print 'shape(gxy_grid) = ', np.shape(gxy_grid)
    # gxy_grid = np.reshape( gxy, (len(xcenters), len(ycenters)))
    
    if saveout:
        title = titlestr = r'$g(x,y)$ with system size=({0:0.3f}'.format(2. * np.min(np.abs(BBox))) + ')'
        save_gxy(xgrid, ygrid, gxy_grid, xc, yc, gxy, infodir, maxgxy=maxgxy, title=title, check=False)

    # Also do a crude g(r) this way
    print 'Computing g(r)...'
    rvals = np.arange(0., gxyL + dr * 0.5, dr)
    gxy_rV = np.sqrt(gxy_xV**2 + gxy_yV**2)
    grV, redges = np.histogram(gxy_rV, bins=rvals)
    rcenters = redges[:-1] + 0.5 * (redges[1:] - redges[:-1])
    grV = grV.astype(float)
    grV *= 1./(density * 2.0 * np.pi * rcenters * dr)
    grV *= (1./float(len_xyip))
    
    if check:
        print 'grV = ',  grV
        print '(1./float(len(xy))) = ', (1./float(len(xy)))
        print 'dif(rcenters) = ',  np.diff(rcenters)
        print 'dr = ',  dr
    
    if saveout:
        save_gr(rcenters, grV, infodir, check=False)
    
    return xgrid, ygrid, gxy_grid, rcenters, grV


def save_gxy(xgrid, ygrid, gxy_grid, xc, yc, gxy, infodir, ax=None, title=r'$g(x,y)$', maxgxy='auto',
             cbar_orientation='vertical', cbar_nticks=2, check=False):
    """Save 2-pt correlation function g(x,y) into infodir"""
    print 'Saving g(x,y) heatmaps...'
    # Save heatmap of g(x,y)
    if ax is None:
        fig, ax, cbar_ax = leplt.initialize_1panel_cbar_fig(wsfrac=0.5)

    if maxgxy == 'auto':
        sca = ax.pcolormesh(xgrid, ygrid, gxy_grid, cmap='viridis', vmin=0.0)
        vmax = np.max(gxy_grid.ravel())
    else:
        vmax = maxgxy
        sca = ax.pcolormesh(xgrid, ygrid, gxy_grid, cmap='viridis', vmin=0.0, vmax=maxgxy)

    cbar = plt.colorbar(sca, cax=cbar_ax, orientation=cbar_orientation)
    cbar.set_ticks([0., vmax*0.5, vmax])
    cbar.set_label(r'$ g(x,y)$', rotation=90)
    ax.axis('equal')
    # print 'title = ', titlestr
    plt.suptitle(title)
    print 'saving figure: ', infodir + 'gxy.png'
    plt.savefig(infodir + 'gxy.png', dpi=600)

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    plt.savefig(infodir + 'gxy_zoom.png', dpi=600)
    if check:
        plt.show()

    # Save g(xy) as txt file
    print 'Saving g(x,y) text file...'
    MM = np.dstack((xc, yc, gxy))[0]
    np.savetxt(infodir + 'gxy.txt', MM, delimiter=',', header='r, g(xy) : correlation function for point set')


def save_gr(rcenters, grV, infodir, title=r'$g(r)$', check=False):
        print 'Saving g(r)...'
        # Plot g(r)
        plt.clf()
        plt.plot(rcenters, grV, 'ro-')
        plt.title(title)
        plt.xlabel(r'$r$')
        plt.ylabel(r'$g(r)$')
        plt.ylim(0.0, np.max(grV)*1.2)
        plt.savefig(infodir + 'gr_crude.png')

        # Plot zoomed g(r)
        plt.xlim(0.0, 5.0)
        plt.savefig(infodir + 'gr_crude_zoom.png')
        if check:
            plt.show()
        plt.clf()

        # Save g(r) as txt file
        MM = np.dstack((rcenters, grV))[0]
        np.savetxt(infodir+'gr_crude.txt', MM, delimiter=',', header='r, g(r) : correlation function for point set')


def calc_fancy_gr(xy, BBox, outdir=None, dr=0.1, check=False):
    """Calculate two-pt correlation function g(r) using all particles in the system, weighting probabilities by areas
    in the bounding box BBox.
    This is much slower than the cruder calc_gxy_gr().
    """
    if outdir is not None:
        infodir = outdir
        saveout = True
        # Register cmaps if not registered, and set colormap
        if not 'viridis' in plt.colormaps():
            plt.register_cmap(name='viridis', cmap=cmaps.viridis)
        plt.set_cmap(cmaps.viridis)
    else:
        saveout = False

    rvals = np.arange(dr, np.mean(np.abs(BBox)) + dr * 0.5, dr)
    # Modify this to calc area of BBox
    # BBox = np.array([[-LL*0.5,-LL*0.5],[LL*0.5,-LL*0.5],[LL*0.5,LL*0.5],[-LL*0.5,LL*0.5]])
    # BBox_sg = sg.Polygon([(-LL*0.5,-LL*0.5),(LL*0.5,-LL*0.5),(LL*0.5,LL*0.5),(-LL*0.5,LL*0.5)])
    BBox_sg = sg.Polygon([tuple(row) for row in BBox])
    print 'BBox_sg = ', BBox_sg
    # Area_check = LL[0] * LL[1]
    Area = BBox_sg.area
    density = float(len(xy))/Area
    print 'density = ', density

    grV = np.zeros(len(rvals), dtype=float)
    # Consider each particle in turn. Histogram distances of other particles.
    # Divide by ((NP-1) * 2 pi r dr), where NP-1 is the # of reference particles we considered.
    # Divide by the particle number density, ensuring g(r)=1 for data with no structure.
    LL0 = np.max(BBox[:, 0]) - np.min(BBox[:, 0])
    for ii in range(len(xy)):
        if np.mod(ii, 50) == 0:
            print 'g(r) particle ', ii
        dists = np.linalg.norm(xy-xy[ii], axis=1)
        hist, bin_edges = np.histogram(dists, bins=len(rvals), range=(dr, LL0 + dr))
        # print 'count = ', hist
        # To get each intersection of the annulus with the bounding box,
        # first form the array of areas for each annulus
        areaV = np.zeros(len(rvals), dtype=float)
        for jj in range(len(rvals)):
            # if pt is distant from BBox edges, use 2.0*np.pi*rvals*dr
            if np.min(abs(xy[ii]-BBox)) > rvals[jj] + dr:
                areaV[jj] = 2.0*np.pi*rvals[jj]*dr
            else:
                rout = sg.Point(xy[ii, 0], xy[ii, 1]).buffer(rvals[jj]+dr)
                rin = sg.Point(xy[ii]).buffer(rvals[jj])
                routA = rout.intersection(BBox_sg).area
                rinA = rin.intersection(BBox_sg).area
                areaV[jj] = routA - rinA

            ######################
            if check:
                # use descartes to create the matplotlib patches
                rout = sg.Point(xy[ii, 0], xy[ii, 1]).buffer(rvals[jj]+dr)
                rin = sg.Point(xy[ii]).buffer(rvals[jj])
                annulus = rout.difference(rin)
                overlap = annulus.intersection(BBox_sg)
                print 'areaV[jj] = ', areaV[jj]
                print 'overlap.area = ', overlap.area
                ax = plt.gca()
                ax.add_patch(descartes.PolygonPatch(annulus, fc='b', ec='k', alpha=0.2))
                ax.add_patch(descartes.PolygonPatch(BBox_sg, fc='r', ec='k', alpha=0.2))
                try:
                    ax.add_patch(descartes.PolygonPatch(overlap, fc='g', ec='k', alpha=0.5))
                except:
                    print 'Could not polygon patch overlap!'
                # control display
                ax.set_xlim(-LL[0]-1, LL[0]+1); ax.set_ylim(-LL[0]-1, LL[0]+1)
                ax.set_aspect('equal')
                plt.show()

            ######################

        keep = np.where(areaV > 1e-6)[0]
        count = (hist[keep]).astype(float)/(float(len(xy))*areaV[keep])  # 2.0*np.pi*rvals*dr)
        # print 'count = ', count
        grV[keep] += count[keep]
        # print 'grV = ', grV

    if check:
        print 'min(areaV)=', np.min(areaV)
        plt.plot(np.arange(len(areaV)), areaV, 'b.-')
        plt.show()
        rout = sg.Point(xy[ii, 0], xy[ii, 1]).buffer(rvals[jj] + dr)
        rin = sg.Point(xy[ii]).buffer(rvals[jj])
        annulus = rout.difference(rin)
        overlap = annulus.intersection(BBox_sg)
        print 'areaV[jj] = ', areaV[jj]
        print 'overlap.area = ', overlap.area

        ax = plt.gca()
        ax.add_patch(descartes.PolygonPatch(annulus, fc='b', ec='k', alpha=0.2))
        ax.add_patch(descartes.PolygonPatch(BBox_sg, fc='r', ec='k', alpha=0.2))
        try:
            ax.add_patch(descartes.PolygonPatch(overlap, fc='g', ec='k', alpha=0.5))
        except:
            print 'Could not polygon patch overlap!'
        # control display
        ax.set_xlim(-LL[0]-1, LL[0]+1)
        ax.set_ylim(-LL[0]-1, LL[0]+1)
        ax.set_aspect('equal')
        plt.show()

    grV /= density
    ######################################################################################################

    if saveout:
        # Plot g(r)
        plt.clf()
        plt.plot(rvals, grV,'ro-')
        plt.title(r'$g(r)$')
        plt.xlabel(r'$r$')
        plt.ylabel(r'$g(r)$')
        plt.ylim(0.0, np.max(grV)*1.2)
        plt.savefig(infodir + 'gr.png')

        # Plot zoomed g(r)
        plt.xlim(0.0, 5.0)
        plt.savefig(infodir + 'gr_zoom.png')

        # Save g(r) as txt file
        MM = np.dstack((rvals, grV))[0]
        np.savetxt(infodir+'gr.txt', MM, delimiter=',', header='r, g(r) : correlation function for point set')

    return rvals, grV
    
   
def pointset_fft(xy, outdir=None, plot_output=False, fftlim=None, check=False):
    """Print an image of the pointset, take an FFT of the image. Returns the 2D and radial power spectrum, and
    plots these if plot_output==True.
    
    Parameters
    ----------
    xy :
    fftlim : tuple of floats
        Limits for ln(power spectrum) values in plotting
    plot_output :
    outdir :
    check : bool
    
    Returns
    ----------
    psf2D :
    psf1D :
    """
    if outdir is None:
        infodir = './'
    else:
        infodir = outdir
        plot_output = True
    
    fig = plt.figure(frameon=False)
    fig.set_size_inches(10,10)
    dpi = 200
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    if len(xy) > 40000:
        ax.scatter(xy[:, 0], xy[:, 1], s=0.5, edgecolor=None, facecolor='k')
    elif len(xy) > 10000:
        ax.scatter(xy[:, 0], xy[:, 1], s=1, edgecolor=None, facecolor='k')
    elif len(xy) > 2000:
        ax.scatter(xy[:, 0], xy[:, 1], s=5, edgecolor=None, facecolor='k')
    else:
        ax.scatter(xy[:, 0], xy[:, 1], s=25, edgecolor=None, facecolor='k')
    fig.add_axes(ax)
    # from SO: ax.imshow(your_image, aspect='normal')
    fname = infodir + 'pointset_for_fft.png'
    fig.savefig(fname, dpi=dpi)
    image = imread(fname, flatten=False, mode='I')
    image[image < 255] = 0
    
    if not plot_output:
        subprocess.call(['rm', fname])
    
    # # Take the fourier transform of the image.
    F1 = fftpack.fft2(image)
    # Now shift the quadrants around so that low spatial frequencies are in
    # the center of the 2D fourier transformed image.
    F2 = fftpack.fftshift( F1 )
    # Calculate a 2D power spectrum
    psf2D = np.abs( F2 )**2
    # Calculate the azimuthally averaged 1D power spectrum
    psf1D = le.azimuthalAverage(psf2D)
     
    if plot_output:
        plt.clf()
        plt.imshow( np.log10(psf2D), cmap='bone')
        # plt.imshow( psf2D )
        if fftlim is not None:
            plt.clim(fftlim[0], fftlim[1])
        ax = plt.gca()
        ax.axis('off')
        plt.savefig(infodir+'pointset_fft_nocb.png')
        
        plt.clf()
        h = plt.imshow( np.log10( psf2D ), cmap='bone')
        # plt.imshow( psf2D )
        if fftlim is not None:
            plt.clim(fftlim[0], fftlim[1])
        cb = plt.colorbar()
        cb.set_label(r'$\log_{10} I$')
        plt.savefig(infodir+'pointset_fft.png')
         
        plt.clf()
        plt.semilogy( psf1D )
        plt.xlabel('Spatial Frequency')
        plt.ylabel('Power Spectrum')
        plt.savefig(infodir+'power_spectrum.png')
    
    return psf2D, psf1D


def pointset(xy, outdir=None, save=True, ax=None, ptsz=None, facecolor='k', dpi=800, **kwargs):
    """Print an image of the pointset. Return the figure and axis

    Parameters
    ----------
    xy : 2d float array
    outdir : path to save image
    save : bool
        Save the image in outdir or pwd
    ax : matplotlib axis instance or None
        Axis on which to draw the points
    ptsz : int
        size of the points to draw (in pixels)
    facecolor : color specifier
        The color of the points to draw
    dpi : int
        Resolution of the image to save, if save==True
    **kwargs : keyword arguments for plotting.initialize_1panel_fig()
    """
    if outdir is None:
        infodir = './'
    else:
        infodir = dio.prepdir(outdir)

    if ax is None:
        fig, ax = leplt.initialize_1panel_fig(**kwargs)
        ax.set_axis_off()

    if ptsz is None:
        ptsz = max(min(25, len(xy)*0.003), 0.5)

    ax.scatter(xy[:, 0], xy[:, 1], s=ptsz, edgecolor=None, facecolor=facecolor)

    if save:
        fname = infodir + 'pointset.png'
        plt.savefig(fname, dpi=dpi)
    else:
        plt.show()


def calc_structure_factor_2d(xy, LL, klim=None, nksteps=50, outdir=None, no_crosshairs=False):
    """Explicitly calculate the structure factor for this lattice at a grid of k values.
    S(k) = 1/N |sum_j exp(-i k.r_j)|^2
    
    Parameters
    ----------
    xy : NP x 2 array
        positions of point set
    LL : tuple
        The system size (bounding box dimensions), assuming system is a square (Xsize, Ysize)
    klim : float or 'auto'
        Maximum value of kx and ky to sample, in units of 2pi/L
    nksteps : int
        number of divisions in |k| to sample (note this is half the number of divisions in k).
    outdir : string or None
        if not None, path to where plots will be saved
    nocrosshairs : bool
        if True, attempts to subtract off the crosshairs near k=0 in the region |k|<2*pi and saves (but does not return)
         results

    Returns
    -----------
    kx, ky : N_kx x N_ky float arrays
        wavevectors in x,y sampled by Skmesh
    Skmesh : N_kx x N_ky float arrays
        Structure factor computed over mesh kx,ky
    Sk : N_kx * N_ky x 1 float array
        the structure
    """
    if outdir is not None:
        infodir = outdir
        saveout = True
        # Register cmaps if not registered, and set colormap
        if not 'viridis' in plt.colormaps():
            plt.register_cmap(name='viridis', cmap=cmaps.viridis)
        
        plt.set_cmap(cmaps.viridis)
    else:
        saveout = False
    
    if klim is None:
        klim = float(LL[0]) * 2.0

    print 'float(LL[0])  = ', float(LL[0])
    kstep = max(int(float(klim)/float(nksteps)), 1)
    kx_tmp0 = 2. * np.pi / float(LL[0]) * np.arange(kstep, klim + kstep*0.9, kstep)
    ky_tmp0 = 2. * np.pi / float(LL[1]) * np.arange(kstep, klim + kstep*0.9, kstep)
    kx_tmp = np.hstack((np.hstack((-kx_tmp0[::-1], np.array([0.]))), kx_tmp0))
    ky_tmp = np.hstack((np.hstack((-ky_tmp0[::-1], np.array([0.]))), ky_tmp0))

    kx, ky = np.meshgrid(kx_tmp, ky_tmp)
    kvecs = np.dstack((kx.ravel(), ky.ravel()))[0]
    Sk = np.zeros(len(kvecs), dtype=complex)
    print 'summing up S(k)...'
    for ii in range(len(kvecs)):
        kvec = kvecs[ii]
        Sk[ii] = np.sum(np.exp(-1j * np.dot(xy[:, 0:2], kvec)))
    # for ii in range(len(kvecs)):
    #     kvec = kvecs[ii]
    #     for pt in xy:
    #         Sk[ii] += np.sum( np.exp(-1j* np.dot(xy[:,0:2]-pt,kvec) ))
    #     print 'summed ', ii, ' of ', len(kvecs)
    
    Sk = (np.abs(Sk)**2/len(xy))
    
    # Subtract off delta function at the center
    origin = np.where(np.logical_and(np.abs(kvecs[:, 0]) == np.min(np.abs(kvecs[:, 0])),
                                     np.abs(kvecs[:, 1]) == np.min(np.abs(kvecs[:, 1]))))[0]
    print '\n\norigin = ', origin, '\n\n'
    Sk[origin] = 0.
    Skmesh = np.reshape(Sk, (len(kx_tmp), len(ky_tmp)))
    # Skmesh[origin] = 0.
    # Sk = np.reshape( Skmesh, (len(kx_tmp)*len(ky_tmp),1)).ravel()

    if saveout:
        # Save linear heatmap of S(k)
        # print '\n\nky= ', ky, '\n\n'
        plt.clf()
        plt.pcolormesh(kx, ky, Skmesh, cmap='viridis', vmin=0.0, vmax=4.0)
        ax = plt.gca()
        cb = plt.colorbar()
        cb.set_label(r'$ S(\mathbf{k})$')
        ax.axis('equal')
        titlestr = r'$S(\mathbf{k})$ with system size='+'({0:0.3f}'.format(LL[0])+', {0:0.3f}'.format(LL[1])+')'
        plt.title(titlestr)
        plt.savefig(infodir + 'Sk.png')
    
        # # Save log heatmap of S(k)
        # plt.clf()
        # plt.pcolor( kx, ky, np.log10( Skmesh ), cmap='viridis', vmin=-4.0, vmax=1.0)
        # titlestr = r'$\log_{10} S(k)$ with system size=('+'{0:0.3f}'.format(LL[0])+', {0:0.3f}'.format(LL[1])+')'
        # print 'titlestr = ', titlestr
        # plt.title(titlestr)
        # ax = plt.gca()
        # cb = plt.colorbar()
        # cb.set_label(r'$\log_{10} S(k)$')
        # ax.axis('equal')
        # plt.savefig(infodir+'Sk_log.png')
        
        # Save S(k) as txt file
        MM = np.dstack((kvecs[:, 0], kvecs[:, 1], Sk))[0]
        np.savetxt(infodir + 'Sk.txt', MM, delimiter=',',
                   header='kx, ky, S(k) : Structure Factor as function of kx and ky for point set')

    if no_crosshairs:
        # Substract off crosshairs
        kmax = np.max(kx)
        bins = np.linspace(0, kmax, nksteps + 1)
        krvec = np.sqrt(kvecs[:, 0] ** 2 + kvecs[:, 1] ** 2)
        ind = np.digitize(krvec, bins)

        # if # rows, # cols in Skmesh is odd, replace center row with average of surrounding ones
        if np.mod(len(Skmesh), 2) == 0:
            print 'Skmesh is even, currently cannot average out the crosshairs!'
        else:
            print 'Skmesh is odd. Use neighboring rows/cols to create crosshair values'
            midIND = int(len(Skmesh) * 0.5 - 0.5)
            in2piX = np.where(np.abs(kx_tmp) < 2. * np.pi * 0.8)[0]
            in2piY = np.where(np.abs(ky_tmp) < 2. * np.pi * 0.8)[0]
            Skmesh[midIND, in2piX] = np.mean(np.vstack((Skmesh[midIND - 1, in2piX], Skmesh[midIND + 1, in2piX])),
                                             axis=0)
            Skmesh[in2piY, midIND] = np.mean(np.vstack((Skmesh[in2piY, midIND - 1], Skmesh[in2piY, midIND + 1])),
                                             axis=0)
        if saveout:
            # Save linear heatmap of S(k)
            plt.clf()
            plt.pcolormesh(kx, ky, Skmesh, cmap='viridis', vmin=0.0, vmax=4.0)
            ax = plt.gca()
            cb = plt.colorbar()
            cb.set_label(r'$ S(\mathbf{k})$')
            ax.axis('equal')
            plt.title(r'$S(k)$ with system size=({0:0.3f}'.format(LL[0]) + ', {0:0.3f}'.format(LL[1]) + ')')
            plt.savefig(infodir + 'Sk_nocrosshair.png')

    return kx, ky, Skmesh, Sk


def calc_structure_factor(xy, LL, klim='auto', nksteps=50, outdir=None, no_crosshairs=False):
    """Explicitly calculate the structure factor for this lattice at a grid of k values.
    S(k) = 1/N |sum_j exp(-i k.r_j)|^2

    Parameters
    ----------
    xy : NP x 2 array
        positions of point set
    LL : tuple
        The system size (bounding box dimensions), assuming system is a square (Xsize, Ysize)
    klim : float or 'auto'
        the upper bound for the magnitude of k vectors probed, in units of 2pi/L
    nksteps : int
        number of divisions in |k| to sample
    outdir : string or None
        if not None, path to where plots will be saved
    no_crosshairs : bool
        if True, attempts to subtract off the crosshairs near k=0 in the region |k|<2*pi and saves (but does not return)
         results

    Returns
    -----------
    kx, ky : N_kx x N_ky float arrays
        wavevectors in x,y sampled by Skmesh
    Skmesh : N_kx x N_ky float arrays
        Structure factor computed over mesh kx,ky
    kr : N x 1 float array
        radial wavevectors
    Skr : N x 1 float array
        Structure factor averaged over magnitude of wavevectors
    """
    if outdir is not None:
        infodir = outdir
        saveout = True
        # Register cmaps if not registered, and set colormap
        if not 'viridis' in plt.colormaps():
            plt.register_cmap(name='viridis', cmap=cmaps.viridis)

        plt.set_cmap(cmaps.viridis)
    else:
        saveout = False

    if klim == 'auto':
        klim = float(LL[0]) * 2.0

    # Get 2D structure factor, to reduce to 1d
    kx, ky, Skmesh, Sk = calc_structure_factor_2d(xy, LL, klim=klim, nksteps=nksteps, outdir=outdir,
                                                  no_crosshairs=False)

    kvecs = np.dstack((kx.ravel(), ky.ravel()))[0]
    kmax = np.max(kvecs[:, 0])

    # Average S(k) over r and average elements each bin
    krvec = np.sqrt(kvecs[:, 0]**2 + kvecs[:, 1]**2)
    bins = np.linspace(0, kmax, nksteps+1)
    ind = np.digitize(krvec, bins)
    kr = 0.5 * (bins[0:-1] + bins[1:])
    Skr = np.zeros(nksteps)
    for ii in range(nksteps):
        try:
            Skr[ii] = np.mean(Sk[ind == ii])
        except:
            print('There are no points to sample in this bin, so skipping it: kr = ' + str(kr[ii]))
        
    if saveout:
        # Plot S(|k|) as profile plot
        plt.clf()
        plt.plot(kr, Skr)
        plt.title('$S(k)$ with system size=({0:0.3f}'.format(LL[0]) + ', {0:0.3f}'.format(LL[1]) + ')')
        plt.xlabel('k')
        plt.ylabel(r'$S(k)$')
        plt.ylim([0, 4.0])
        plt.xlim([0, kmax])
        plt.savefig(infodir + 'Skr.png')
        
    if no_crosshairs:
        # Substract off crosshairs
        # if # rows, # cols in Skmesh is odd, replace center row with average of surrounding ones
        if np.mod(len(Skmesh), 2) == 0:
            print 'Skmesh is even, currently cannot average out the crosshairs!'
        else:
            print 'Skmesh is odd.'
            kstep = klim / float(nksteps)
            kx_tmp = 2. * np.pi/LL[0] * np.arange(-klim, klim + kstep*0.9, kstep)
            ky_tmp = 2. * np.pi/LL[1] * np.arange(-klim, klim + kstep*0.9, kstep)
            midIND = int(len(Skmesh)*0.5-0.5)
            in2piX = np.where(np.abs(kx_tmp) < 2. * np.pi * 0.8)[0]
            in2piY = np.where(np.abs(ky_tmp) < 2. * np.pi * 0.8)[0]
            Skmesh[midIND, in2piX] = np.mean(np.vstack((Skmesh[midIND-1, in2piX], Skmesh[midIND+1, in2piX])), axis=0)
            Skmesh[in2piY, midIND] = np.mean(np.vstack((Skmesh[in2piY, midIND-1], Skmesh[in2piY, midIND+1])), axis=0)
            
            # REMAKE Sk with no crosshairs (call it Sknch for Sk-no-crosshairs)
            Sknch = np.reshape(Skmesh, (len(kx_tmp)*len(ky_tmp), 1)).ravel()
            
            # Average S(k) over r and average elements each bin
            Skrnch = np.zeros(nksteps)
            for ii in range(nksteps):
                Skrnch[ii] = np.mean(Sknch[ind == ii])
            
        if saveout:
            # Plot S(|k|) as profile plot
            plt.clf()
            plt.plot(kr, Skrnch)
            plt.title('$S(k)$ with system size=({0:0.3f}'.format(LL[0]) + ', {0:0.3f}'.format(LL[1]) + ')')
            plt.xlabel('k')
            plt.ylabel(r'$S(k)$')
            plt.ylim([0, 4.0])
            plt.xlim([0, kmax])
            plt.savefig(infodir + 'Skr_nocrosshair.png')

    # print 'np.shape(kx) = ', np.shape(kx)
    # print 'np.shape(ky) = ', np.shape(ky)
    # print 'np.shape(Skmesh) = ', np.shape(Skmesh)
    # print 'np.shape(kr) = ', np.shape(kr)
    # print 'np.shape(Skr) = ', np.shape(Skr)
    return kx, ky, Skmesh, kr, Skr


if __name__ == '__main__':
    # Demonstrate S(k) calc
    plt.register_cmap(name='viridis', cmap=cmaps.viridis)
    plt.set_cmap(cmaps.viridis)

    # Create hy
    sz = 100.0
    xx = np.arange(sz, dtype=float)
    xy = np.dstack((np.meshgrid(xx, xx)[0].ravel(), np.meshgrid(xx, xx)[1].ravel()))[0]
    xy = xy + np.random.rand(len(xy), 2)
    plt.scatter(xy[:, 0], xy[:, 1], c='k', s=1)
    plt.axis('equal')
    plt.title('Uniformly-randomized square lattice')
    plt.show()
    LL = (sz, sz)
    kx, ky, Skmesh, kr, Skr = calc_structure_factor(xy, LL, klim=LL[0], nksteps=50, outdir=None, no_crosshairs=False)
    plt.loglog(kr, Skr, '.-')
    plt.loglog(kr, kr**2, '--')
    plt.xlabel(r'wavenumber $k$')
    plt.ylabel(r'$S(k)$')
    plt.title(r'$S(k)$ for uniformly-randomized square lattice ({0:0.3f}'.format(LL[0]) +
              ', {0:0.3f}'.format(LL[1]) + ')')
    plt.show()
    plt.pcolormesh(kx, ky, Skmesh, cmap='viridis', vmin=0.0, vmax=4.0)
    plt.sca(plt.gca())
    cb = plt.colorbar()
    cb.set_label(r'$ S(\mathbf{k})$')
    plt.axis('equal')
    plt.title(r'$S(k)$ for uniformly-randomized square lattice ({0:0.3f}'.format(LL[0]) +
              ', {0:0.3f}'.format(LL[1]) + ')')
    plt.show()

    xx = np.arange(sz, dtype=float)
    xy = np.dstack((np.meshgrid(xx, xx)[0].ravel(), np.meshgrid(xx, xx)[1].ravel()))[0]
    xy = xy + np.random.normal(size=(len(xy), 2))
    plt.scatter(xy[:, 0], xy[:, 1], c='k', s=1)
    plt.axis('equal')
    plt.title('Gaussian-randomized square lattice')
    plt.show()
    LL = (sz, sz)
    kx, ky, Skmesh, kr, Skr = calc_structure_factor(xy, LL, klim=LL[0], nksteps=50, outdir=None, no_crosshairs=False)
    plt.loglog(kr, Skr, '.-')
    plt.loglog(kr, kr**2, '--')
    plt.xlabel(r'wavenumber $k$')
    plt.ylabel(r'$S(k)$')
    plt.title(r'$S(k)$ for Gaussian-randomized square lattice ({0:0.2f}'.format(LL[0]) +
              ', {0:0.2f}'.format(LL[1]) + ')')
    plt.show()
    plt.pcolormesh(kx, ky, Skmesh, cmap='viridis', vmin=0.0, vmax=4.0)
    plt.axis('equal')
    cb = plt.colorbar()
    cb.set_label(r'$ S(\mathbf{k})$')
    plt.axis('equal')
    plt.title(r'$S(k)$ for Gaussian-randomized square lattice ({0:0.2f}'.format(LL[0]) +
              ', {0:0.2f}'.format(LL[1]) + ')')
    plt.show()

