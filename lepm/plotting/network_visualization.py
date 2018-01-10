import numpy as np
import lepm.dataio
import matplotlib.pyplot as plt
import lepm.lattice_elasticity as le
import lepm.plotting.colormaps as lecmaps
import lepm.plotting.plotting as leplt
from matplotlib.collections import LineCollection
from matplotlib.collections import PatchCollection


'''Functions which are of more general use than time domain applications'''


def movie_plot_2D(xy, BL, bs=None, fname='none', title='', NL=[], KL=[], BLNNN=[], NLNNN=[], KLNNN=[], PVx=[],
                  PVy=[], PVxydict={}, nljnnn=None, kljnnn=None, klknnn=None,
                  ax=None, fig=None, axcb='auto', cbar_ax=None, cbar_orientation='vertical',
                  xlimv='auto', ylimv='auto', climv=0.1, colorz=True, ptcolor=None, figsize='auto', colorpoly=False,
                  bondcolor=None, colormap='seismic', bgcolor=None, axis_off=False,
                  axis_equal=True, text_topleft=None, lw=-1.,
                  ptsize=10, negative_NNN_arrows=False, show=False, arrow_alpha=1.0,
                  fontsize=8, cax_label='Strain', zorder=0, rasterized=False):
    """Plots (and saves if fname is not 'none') a 2D image of the lattice with colored bonds and particles colored by
    coordination (both optional).

    Parameters
    ----------
    xy : array of dimension nx2
        2D lattice of points (positions x,y)
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points
    bs : array of dimension #bonds x 1 or None
        Strain in each bond
    fname : string
        Full path including name of the file (.png, etc), if None, will not save figure
    title : string
        The title of the frame
    NL : NP x NN int array (optional, for speed)
        Specify to speed up computation, if colorz or colorpoly or if periodic boundary conditions
    KL : NP x NN int array (optional, for speed)
        Specify to speed up computation, if colorz or colorpoly or if periodic boundary conditions
    BLNNN :
    NLNNN :
    KLNNN :
    PVx : NP x NN float array (optional, for periodic lattices and speed)
        ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen by particle i
        If PVx and PVy are specified, PVxydict need not be specified.
    PVy : NP x NN float array (optional, for periodic lattices and speed)
        ijth element of PVy is the y-component of the vector taking NL[i,j] to its image as seen by particle i
        If PVx and PVy are specified, PVxydict need not be specified.
    PVxydict : dict (optional, for periodic lattices)
        dictionary of periodic bonds (keys) to periodic vectors (values)
    nljnnn : #pts x max(#NNN) int array or None
        nearest neighbor array matching NLNNN and KLNNN. nljnnn[i, j] gives the neighbor of i such that NLNNN[i, j] is
        the next nearest neighbor of i through the particle nljnnn[i, j]
    kljnnn : #pts x max(#NNN) int array or None
        bond array describing periodicity of bonds matching NLNNN and KLNNN. kljnnn[i, j] describes the bond type
        (bulk -> +1, periodic --> -1) of bond connecting i to nljnnn[i, j]
    klknnn : #pts x max(#NNN) int array or None
        bond array describing periodicity of bonds matching NLNNN and KLNNN. klknnn[i, j] describes the bond type
        (bulk -> +1, periodic --> -1) of bond connecting nljnnn[i, j] to NLNNN[i, j]
    ax: matplotlib figure axis instance
        Axis on which to draw the network
    fig: matplotlib figure instance
        Figure on which to draw the network
    axcb: matplotlib colorbar instance
        Colorbar to use for strains in bonds
    cbar_ax : axis instance
        Axis to use for colorbar. If colorbar instance is not already defined, use axcb instead.
    cbar_orientation : str ('horizontal' or 'vertical')
        Orientation of the colorbar
    xlimv: float or tuple of floats
    ylimv: float or tuple of floats
    climv : float or tuple
        Color limit for coloring bonds by bs
    colorz: bool
        whether to color the particles by their coordination number
    ptcolor: string color spec or tuple color spec or None
        color specification for coloring the points, if colorz is False. Default is None (no coloring of points)
    figsize : tuple
        w,h tuple in inches
    colorpoly : bool
        Whether to color in polygons formed by bonds according to the number of sides
    bondcolor : color specification (hexadecimal or RGB)
    colormap : if bondcolor is None, uses bs array to color bonds
    bgcolor : hex format string, rgb color spec, or None
        If not None, sets the bgcolor. Often used is '#d9d9d9'
    axis_off : bool
        Turn off the axis border and canvas
    axis_equal : bool
    text_topleft : str or None
    lw: float
        line width for plotting bonds. If lw == -1, then uses automatic line width to adjust for bond density..
    ptsize: float
        size of points passed to absolute_sizer
    negative_NNN_arrows : bool
        make positive and negative NNN hoppings different color
    show : bool
        whether to show the plot after creating it
    arrow_alpha : float
        opacity of the arrow
    fontsize : int (default=8)
        fontsize for all labels
    cax_label : int (default='Strain')
        Label for the colorbar
    zorder : int
        z placement on axis (higher means bringing network to the front, lower is to the back

    Returns
    ----------
    [ax,axcb] : stuff to clear after plotting

    """
    if fig is None or fig == 'none':
        fig = plt.gcf()
    if ax is None or ax == 'none':
        if figsize == 'auto':
            plt.clf()
        else:
            fig = plt.figure(figsize=figsize)
        ax = plt.axes()

    if bs is None:
        bs = np.zeros_like(BL[:, 0], dtype=float)

    if colormap not in plt.colormaps():
        lecmaps.register_colormaps()

    NP = len(xy)
    if lw == -1:
        if NP < 10000:
            lw = 0.5
            s = leplt.absolute_sizer()
        else:
            lw = (10/np.sqrt(len(xy)))

    if NL == [] and KL == []:
        if colorz or colorpoly:
            NL, KL = le.BL2NLandKL(BL, NP=NP, NN='min')
            if (BL < 0).any():
                if len(PVxydict) == 0:
                    raise RuntimeError('PVxydict must be supplied to display_lattice_2D() when periodic BCs exist, ' +
                                       'if NL and KL not supplied!')
                else:
                    PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL, KL)

    if colorz:
        zvals = (KL != 0).sum(1)
        zmed = np.median(zvals)
        # print 'zmed = ', zmed
        under1 = np.logical_and(zvals < zmed-0.5, zvals > zmed-1.5)
        over1 = np.logical_and(zvals > zmed+0.5, zvals < zmed+1.5)
        Cz = np.zeros((len(xy), 3), dtype=int)
        # far under black // under blue // equal white // over red // far over green
        Cz[under1] = [0./255, 50./255, 255./255]
        Cz[zvals == zmed] = [100./255, 100./255, 100./255]
        Cz[over1] = [255./255, 0./255, 0./255]
        Cz[zvals > zmed+1.5] = [0./255, 255./255, 50./255]
        # Cz[zvals<zmed-1.5] = [0./255,255./255,150./255] #leave these black

        s = leplt.absolute_sizer()
        sval = min([.005, .12/np.sqrt(len(xy))])
        sizes = np.zeros(NP, dtype=float)
        sizes[zvals > zmed + 0.5] = sval
        sizes[zvals == zmed] = sval * 0.5
        sizes[zvals < zmed - 0.5] = sval

        # topinds = zvals!=zmed
        ax.scatter(xy[:, 0], xy[:, 1], s=s(sizes), c=Cz, edgecolor='none', zorder=10, rasterized=rasterized)
        ax.axis('equal')
    elif ptcolor is not None and ptcolor != 'none' and ptcolor != '':
        if NP < 10000:
            # if smallish #pts, plot them
            # print 'xy = ', xy
            # plt.plot(xy[:,0],xy[:,1],'k.')
            s = leplt.absolute_sizer()
            ax.scatter(xy[:, 0], xy[:, 1], s=ptsize, alpha=0.5, facecolor=ptcolor, edgecolor='none',
                       rasterized=rasterized)

    if colorpoly:
        # Color the polygons based on # sides
        # First extract polygons. To do that, if there are periodic boundaries, we need to supply as dict
        if PVxydict == {} and len(PVx) > 0:
            PVxydict = le.PVxy2PVxydict(PVx, PVy, NL, KL=KL)

        polygons = le.extract_polygons_lattice(xy, BL, NL=NL, KL=KL, viewmethod=True, PVxydict=PVxydict)
        PolyPC = le.polygons2PPC(polygons)
        # number of polygon sides
        Pno = np.array([len(polyg) for polyg in polygons], dtype=int)
        print 'nvis: Pno = ', Pno
        print 'nvis: medPno = ', np.floor(np.median(Pno))
        medPno = np.floor(np.median(Pno))
        uIND = np.where(Pno == medPno-1)[0]
        mIND = np.where(Pno == medPno)[0]
        oIND = np.where(Pno == medPno+1)[0]
        loIND = np.where(Pno < medPno-1.5)[0]
        hiIND = np.where(Pno > medPno+1.5)[0]
        print ' uIND = ', uIND
        print ' oIND = ', oIND
        print ' loIND = ', loIND
        print ' hiIND = ', hiIND
        if len(uIND) > 0:
            PPCu = [PolyPC[i] for i in uIND]
            pu = PatchCollection(PPCu, color='b', alpha=0.5)
            ax.add_collection(pu)
        if len(mIND) > 0:
            PPCm = [PolyPC[i] for i in mIND]
            pm = PatchCollection(PPCm, color=[0.5, 0.5, 0.5], alpha=0.5)
            ax.add_collection(pm)
        if len(oIND) > 0:
            PPCo = [PolyPC[i] for i in oIND]
            po = PatchCollection(PPCo, color='r', alpha=0.5)
            ax.add_collection(po)
        if len(loIND) > 0:
            PPClo = [PolyPC[i] for i in loIND]
            plo = PatchCollection(PPClo, color='k', alpha=0.5)
            ax.add_collection(plo)
        if len(hiIND) > 0:
            PPChi = [PolyPC[i] for i in hiIND]
            phi = PatchCollection(PPChi, color='g', alpha=0.5)
            ax.add_collection(phi)

    # Efficiently plot many lines in a single set of axes using LineCollection
    # First check if there are periodic bonds
    if BL.size > 0:
        if (BL < 0).any():
            if PVx == [] or PVy == [] or PVx is None or PVy is None:
                raise RuntimeError('PVx and PVy must be supplied to display_lattice_2D when periodic BCs exist!')
            else:
                # get indices of periodic bonds
                perINDS = np.unique(np.where(BL < 0)[0])
                perBL = np.abs(BL[perINDS])
                # # Check
                # print 'perBL = ', perBL
                # plt.plot(xy[:,0], xy[:,1],'b.')
                # for i in range(len(xy)):
                #     plt.text(xy[i,0]+0.05, xy[i,1],str(i))
                # plt.show()

                # define the normal bonds which are not periodic
                normINDS = np.setdiff1d(np.arange(len(BL)), perINDS)
                BLtmp = BL[normINDS]
                bstmp = bs[normINDS]
                lines = [zip(xy[BLtmp[i, :], 0], xy[BLtmp[i, :], 1]) for i in range(len(BLtmp))]

                xy_add = np.zeros((4, 2))
                # Build new strain list bs_out by storing bulk lines first, then recording the strain twice
                # for each periodic bond since the periodic bond is at least two lines in the plot, get bs_out
                # ready for appending
                # bs_out = np.zeros(len(normINDS) + 5 * len(perINDS), dtype=float)
                # bs_out[0:len(normINDS)] = bstmp
                bs_out = bstmp.tolist()

                # Add periodic bond lines to image
                # Note that we have to be careful that a single particle can be connected to another both in the bulk
                # and through a periodic boundary, and perhaps through more than one periodic boundary
                # kk indexes perINDS to determine what the strain of each periodic bond should be
                # draw_perbond_count counts the number of drawn linesegments that are periodic bonds
                kk, draw_perbond_count = 0, 0
                for row in perBL:
                    colA = np.argwhere(NL[row[0]] == row[1]).ravel()
                    colB = np.argwhere(NL[row[1]] == row[0]).ravel()
                    if len(colA) > 1 or len(colB) > 1:
                        # Look for where KL < 0 to pick out just the periodic bond(s) -- ie there
                        # were both bulk and periodic bonds connecting row[0] to row[1]
                        a_klneg = np.argwhere(KL[row[0]] < 0)
                        colA = np.intersect1d(colA, a_klneg)
                        b_klneg = np.argwhere(KL[row[1]] < 0)
                        colB = np.intersect1d(colB, b_klneg)
                        # print 'colA = ', colA
                        # print 'netvis here'
                        # sys.exit()
                        if len(colA) > 1 or len(colB) > 1:
                            # there are multiple periodic bonds connecting one particle to another (in different
                            # directions). Plot each of them.
                            for ii in range(len(colA)):
                                print 'netvis: colA = ', colA
                                print 'netvis: colB = ', colB
                                # columns a and b for this ii index
                                caii, cbii = colA[ii], colB[ii]
                                # add xy points to the network to plot to simulate image particles
                                xy_add[0] = xy[row[0]]
                                xy_add[1] = xy[row[1]] + np.array([PVx[row[0], caii], PVy[row[0], caii]])
                                xy_add[2] = xy[row[1]]
                                xy_add[3] = xy[row[0]] + np.array([PVx[row[1], cbii], PVy[row[1], cbii]])
                                # Make the lines to draw (dashed lines for this periodic case)
                                lines += zip(xy_add[0:2, 0], xy_add[0:2, 1]), zip(xy_add[2:4, 0], xy_add[2:4, 1])
                                bs_out.append(bs[perINDS[kk]])
                                bs_out.append(bs[perINDS[kk]])
                                draw_perbond_count += 1

                            kk += 1
                        else:
                            # print 'row = ', row
                            # print 'NL = ', NL
                            # print 'KL = ', KL
                            # print 'colA, colB = ', colA, colB
                            colA, colB = colA[0], colB[0]
                            xy_add[0] = xy[row[0]]
                            xy_add[1] = xy[row[1]] + np.array([PVx[row[0], colA], PVy[row[0], colA]])
                            xy_add[2] = xy[row[1]]
                            xy_add[3] = xy[row[0]] + np.array([PVx[row[1], colB], PVy[row[1], colB]])
                            lines += zip(xy_add[0:2, 0], xy_add[0:2, 1]), zip(xy_add[2:4, 0], xy_add[2:4, 1])
                            # bs_out[2 * kk + len(normINDS)] = bs[perINDS[kk]]
                            # bs_out[2 * kk + 1 + len(normINDS)] = bs[perINDS[kk]]
                            bs_out.append(bs[perINDS[kk]])
                            bs_out.append(bs[perINDS[kk]])
                            draw_perbond_count += 1
                            kk += 1
                    else:
                        colA, colB = colA[0], colB[0]
                        xy_add[0] = xy[row[0]]
                        xy_add[1] = xy[row[1]] + np.array([PVx[row[0], colA], PVy[row[0], colA]])
                        xy_add[2] = xy[row[1]]
                        xy_add[3] = xy[row[0]] + np.array([PVx[row[1], colB], PVy[row[1], colB]])
                        lines += zip(xy_add[0:2, 0], xy_add[0:2, 1]), zip(xy_add[2:4, 0], xy_add[2:4, 1])
                        # bs_out[2 * kk + len(normINDS)] = bs[perINDS[kk]]
                        # bs_out[2 * kk + 1 + len(normINDS)] = bs[perINDS[kk]]
                        bs_out.append(bs[perINDS[kk]])
                        bs_out.append(bs[perINDS[kk]])
                        draw_perbond_count += 1
                        kk += 1

                # replace bs by the new bs (bs_out)
                bs = np.array(bs_out)
                # store number of bulk bonds
                nbulk_bonds = len(normINDS)
        else:
            if len(np.shape(BL)) > 1:
                lines = [zip(xy[BL[i, :], 0], xy[BL[i, :], 1]) for i in range(np.shape(BL)[0])]
                # store number of bulk bonds
                nbulk_bonds = len(lines)
            else:
                lines = [zip(xy[BL[i][0]], xy[BL[i][1]]) for i in range(np.shape(BL)[0])]
                # store number of bulk bonds
                nbulk_bonds = 1

        if isinstance(climv, tuple):
            cmin = climv[0]
            cmax = climv[1]
        elif isinstance(climv, float):
            cmin = -climv
            cmax = climv
        elif climv is None:
            cmin = None
            cmax = None

        if bondcolor is None:
            # draw the periodic bonds as dashed, regular bulk bonds as solid
            line_segments = LineCollection(lines[0:nbulk_bonds],  # Make a sequence of x,y pairs
                                           linewidths=lw,  # could iterate over list
                                           linestyles='solid',
                                           cmap=colormap,
                                           norm=plt.Normalize(vmin=cmin, vmax=cmax),
                                           zorder=zorder, rasterized=rasterized)
            line_segments.set_array(bs[0:nbulk_bonds])
            # draw the periodic bonds as dashed, if there are any
            periodic_lsegs = LineCollection(lines[nbulk_bonds:],  # Make a sequence of x,y pairs
                                            linewidths=lw,  # could iterate over list
                                            linestyles='dashed',
                                            cmap=colormap,
                                            norm=plt.Normalize(vmin=cmin, vmax=cmax),
                                            zorder=zorder, rasterized=rasterized)
            periodic_lsegs.set_array(bs[nbulk_bonds:])
        else:
            line_segments = LineCollection(lines[0:nbulk_bonds], linewidths=lw, linestyles='solid', colors=bondcolor,
                                            zorder=zorder, rasterized=rasterized)
            # draw the periodic bonds as dashed, if there are any
            periodic_lsegs = LineCollection(lines[nbulk_bonds:], linewidths=lw, linestyles='dashed', colors=bondcolor,
                                            zorder=zorder, rasterized=rasterized)

        ax.add_collection(line_segments)
        if periodic_lsegs:
            ax.add_collection(periodic_lsegs)
        # If there is only a single bond color, ignore the colorbar specification
        if bondcolor is None or isinstance(bondcolor, np.ndarray):
            if axcb == 'auto':
                if cbar_ax is None:
                    print 'nvis: Instantiating colorbar...'
                    axcb = fig.colorbar(line_segments)
                else:
                    print 'nvis: Using cbar_ax to instantiate colorbar'
                    axcb = fig.colorbar(line_segments, cax=cbar_ax, orientation=cbar_orientation)

            if axcb != 'none' and axcb is not None:
                print 'nvis: Creating colorbar...'
                axcb.set_label(cax_label, fontsize=fontsize)
                axcb.set_clim(vmin=cmin, vmax=cmax)
        else:
            # Ignore colorbar axis specification
            axcb = 'none'
    else:
        axcb = 'none'

    if len(BLNNN) > 0:
        # todo: add functionality for periodic NNN connections
        # Efficiently plot many lines in a single set of axes using LineCollection
        lines = [zip(xy[BLNNN[i, :], 0], xy[BLNNN[i, :], 1]) for i in range(len(BLNNN))]
        linesNNN = LineCollection(lines,  # Make a sequence of x,y pairs
                                  linewidths=lw,  # could iterate over list
                                  linestyles='dashed',
                                  color='blue',
                                  zorder=100)
        ax.add_collection(linesNNN, rasterized=rasterized)
    elif len(NLNNN) > 0 and len(KLNNN) > 0:
        factor = 0.8
        if (BL < 0).any():
            print 'nvis: plotting periodic NNN...'
            if nljnnn is None:
                raise RuntimeError('Must supply nljnnn to plot NNN hoppings/connections')
            for i in range(NP):
                todo = np.where(KLNNN[i, :] > 1e-12)[0]
                for index in todo:
                    kk = NLNNN[i, index]
                    # Ascribe the correct periodic vector based on both PVx[i, NNind] and PVx[NNind, ind]
                    # Note : nljnnn is
                    # nearest neighbor array matching NLNNN and KLNNN. nljnnn[i, j] gives the neighbor of i such that
                    # NLNNN[i, j] is the next nearest neighbor of i through the particle nljnnn[i, j]
                    jj = nljnnn[i, index]
                    if kljnnn[i, index] < 0 or klknnn[i, index] < 0:
                        jind = np.where(NL[i, :] == jj)[0][0]
                        kind = np.where(NL[jj, :] == kk)[0][0]
                        # print 'jj = ', jj
                        # print 'kk = ', kk
                        # print 'jind = ', jind
                        # print 'kind = ', kind
                        # print 'NL[i, :] =', NL[i, :]
                        # print 'NL[jj, :] =', NL[jj, :]
                        # print 'PVx[i, jind] = ', PVx[i, jind]
                        # print 'PVy[i, jind] = ', PVy[i, jind]
                        # print 'PVx[jj, kind] = ', PVx[jj, kind]
                        # print 'PVy[jj, kind] = ', PVy[jj, kind]
                        dx = (xy[kk, 0] + PVx[i, jind] + PVx[jj, kind] - xy[i, 0]) * factor
                        dy = (xy[kk, 1] + PVy[i, jind] + PVy[jj, kind] - xy[i, 1]) * factor
                    else:
                        dx = (xy[kk, 0] - xy[i, 0]) * factor
                        dy = (xy[kk, 1] - xy[i, 1]) * factor
                    ax.arrow(xy[i, 0], xy[i, 1], dx, dy, head_width=0.1, head_length=0.2, fc='b', ec='b',
                             linestyle='dashed')
                    # Check
                    # print 'dx = ', dx
                    # print 'dy = ', dy
                    # for ind in range(NP):
                    #     plt.text(xy[ind, 0]-0.2, xy[ind, 1]-0.2, str(ind))
                    # plt.show()
                    # sys.exit()
        else:
            # amount to offset clockwise nnn arrows
            for i in range(NP):
                todo = np.where(KLNNN[i, :] > 1e-12)[0]

                # Allow for both blue and red arrows (forward/backward), or just blue. If just blue, use no offset and
                # full scale factor
                if negative_NNN_arrows:
                    scalef = 0.3
                else:
                    scalef = 0.8
                offset = np.array([0.0, 0.0])
                for ind in NLNNN[i, todo]:
                    if negative_NNN_arrows:
                        offset = (xy[ind, :]-xy[i, :])*0.5
                    ax.arrow(xy[i, 0] + offset[0], xy[i, 1] + offset[1], (xy[ind, 0] - xy[i, 0]) * scalef,
                             (xy[ind, 1]-xy[i, 1])*scalef, head_width=0.1, head_length=0.2, fc='b', ec='b',
                             alpha=arrow_alpha)

                if negative_NNN_arrows:
                    todo = np.where(KLNNN[i, :] < -1e-12)[0]
                    for ind in NLNNN[i, todo]:
                        offset = (xy[ind, :] - xy[i, :])*0.5
                        ax.arrow(xy[i, 0] + offset[0], xy[i, 1] + offset[1], (xy[ind, 0] - xy[i, 0]) * 0.3,
                                 (xy[ind, 1]-xy[i, 1]) * 0.3, head_width=0.1, head_length=0.2, fc='r', ec='r',
                                 alpha=arrow_alpha)

    if bgcolor is not None:
        ax.set_axis_bgcolor(bgcolor)

    # set limits
    ax.axis('scaled')
    if xlimv != 'auto' and xlimv is not None:
        if isinstance(xlimv, tuple):
            ax.set_xlim(xlimv[0], xlimv[1])
        else:
            print 'nvis: setting xlimv'
            ax.set_xlim(-xlimv, xlimv)
    else:
        ax.set_xlim(np.min(xy[:, 0]) - 2.5, np.max(xy[:, 0]) + 2.5)

    if ylimv != 'auto' and ylimv is not None:
        if isinstance(ylimv, tuple):
            print 'nvis: setting ylimv to tuple'
            ax.set_ylim(ylimv[0], ylimv[1])
        else:
            ax.set_ylim(-ylimv, ylimv)
    else:
        print 'nvis: setting', min(xy[:, 1]), max(xy[:, 1])
        ax.set_ylim(np.min(xy[:, 1]) - 2, np.max(xy[:, 1]) + 2)

    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    if text_topleft is not None:
        ax.text(0.05, .98, text_topleft,
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes)
    if axis_off:
        ax.axis('off')

    if fname != 'none' and fname != '' and fname is not None:
        print 'nvis: saving figure: ', fname
        plt.savefig(fname)
    if show:
        plt.show()

    return [ax, axcb]

