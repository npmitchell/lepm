import numpy as np
import subprocess
import glob
import matplotlib.pyplot as plt
import lepm.plotting.plotting as leplt
import lepm.lattice_elasticity as le
import lepm.dataio as dio
import copy
import cPickle

'''Auxiliary and low-level functions for plotting time domain simulations'''


def le_plot_gyros(xy, xy0, NL, KL, BM, params, t, ii, name, fig, ax, outdir, climv='auto', exaggerate=1.0,
                  dpi=300, fontsize=12, title='', axis_off=False, **kwargs):
    """Plots a single gyroscopic lattice time step using timestep plot.

    Parameters
    ----------
    t : float
        time stamp for image
    ii : int
        index to name file ( ie 'name_000ii.png')
    name : string
        the name of the file (before _index.png)
    fig : matplotlib.pyplot figure handle, or 'none'
        the figure on which to plot the gyros. If 'none', uses plt.gcf() and clears figure.
    ax : matplotlib.pyplot axis handle, or 'none'
        the axis on which to plot the gyros. If 'none', uses plt.gca()
    outdir : string
        The output directory for the image
    climv : float or tuple
        Color limit for coloring bonds by bond strain, overriddes params['climv'] if climv!='auto'
        If 'climv' is not a key in params, and climv=='auto', then uses default min/max.
        If 'climv' is a key in params, and climv=='auto', then uses params['climv'].
    exaggerate : float (default 1.0 --> in which case it is ignored)
        Exaggerate the displacements of each particle from its initial position by this factor. Ignored if == 1.0
    dpi : float
        pixels per inch, resolution of saved plot
    **kwargs: Additional timestep_plot() keyword arguments
        color_particles='k', fontsize=14, linewidth=2

    Returns
    ----------
    """
    # If fig and ax are not supplied, declare them
    if fig is None or fig == 'none':
        fig = plt.gcf()
        plt.clf()
    if ax is None or ax == 'none':
        ax = plt.gca()

    # make output dir
    outdir = dio.prepdir(outdir)
    dio.ensure_dir(outdir)
    # set range of window from first values
    if 'xlimv' in params:
        xlimv = params['xlimv']
        if 'ylimv' in params:
            ylimv = params['ylimv']
        else:
            ylimv = (xlimv - max(xy0[:, 0])) + np.ceil(np.max(np.abs(xy0[:, 1])))
            # print 'ylimv is calced from xlimv'
            # sys.exit()
    else:
        xlimv = np.ceil(max(xy0[:, 0]) * 5. / 4.)
        ylimv = (xlimv - max(xy0[:, 0])) + max(np.abs(xy0[:, 1]))

    # save current data as stills
    index = '{0:08d}'.format(ii)
    outname = outdir + '/' + name + '_' + index + '.png'
    BL = le.NL2BL(NL, KL)
    # print 'BL = ', BL

    suptitle = copy.deepcopy(title)

    if 'prestrain' in params:
        prestrain = params['prestrain']
    else:
        prestrain = 0.

    if 'shrinkrate' in params:
        shrinkrate = params['shrinkrate']
        title = 't = ' + '%07.0f' % t + '  a = ' + '%07.5f' % (1. - shrinkrate * t - prestrain)
    elif 'prestrain' in params:
        shrinkrate = 0.0
        title = 't = ' + '%07.0f' % t + '  a = ' + '%07.5f' % (1. - prestrain)
    else:
        shrinkrate = 0.0
        title = 't = ' + '%07.0f' % t + r' $\Omega_g^{-1}$'

    if exaggerate != 1.0:
        title += '   amplified ' + str(int(exaggerate)) + 'x  '

    om_nonstandard = False
    one_is_zero = False
    if 'Omk' in params and params['Omk'] != -1.:
        om_nonstandard = True
        if params['Omk'] == 0.:
            one_is_zero = True

    if 'Omg' in params and params['Omk'] != -1.:
        om_nonstandard = True
        if params['Omg'] == 0.:
            one_is_zero = True

    if om_nonstandard and not one_is_zero:
        # check if the ratio is an integer
        if (params['Omk'] / params['Omg']) % 1 < 1e-7:
            # ratio of Omk/Omg is integer
            title += r'   $\Omega_k/\Omega_g$=' + '{0:0d}'.format(int(params['Omk']/params['Omg']))
        else:
            # ratio is not integer
            title += r'   $\Omega_k$=' + '{0:.3f}'.format(params['Omk'] / params['Omg'])
    elif om_nonstandard:
        title += '\n' + r'$\Omega_g$=' + '{0:.3f}'.format(params['Omg'])
        title += r'   $\Omega_k$=' + '{0:.3f}'.format(params['Omk'])

    if 'split_spin' in params:
        title += ' NV=' + str(int(params['NV']))
        if 'split_k' in params:
            title += r'   $k_s$=' + str(params['split_k'])

    # title +='\n'

    # if params['BCtype'] == 'excite':
    #     title += r'   $\omega_d$ = ' + '{0:.3f}'.format(params['frequency'])

    if 'eta' in params:
        if params['eta'] != 0.000:
            title += r'   $\eta$=' + '{0:.3f}'.format(params['eta'])

            # calculate strain
            # bs = bond_strain_list(xy,BL,bL0)

            # if exaggerate==1.0:
            # movie_plot_2D_gyros(xy, BL, bs, outname, title, xlimv, ylimv, climv)
            # else:
            # xye = xy0+(xy-xy0)*exaggerate
            # movie_plot_2D_gyros(xye, BL, bs, outname, title, xlimv, ylimv, climv)

    if climv == 'auto':
        if 'climv' in params:
            climv = params['climv']

    # plt.show()
    # print 'tdgyros: here'
    # sys.exit()
    [scat_fg, lines_st, p] = leplt.timestep_plot(xy, xy0, NL, KL, BM, ax=ax, factor=exaggerate, amp=climv, title=title,
                                                 fontsize=fontsize, suptitle=suptitle, **kwargs)

    # print 'color_particles = ', scat_fg
    ax.set_xlim(-xlimv, xlimv)
    ax.set_ylim(-ylimv, ylimv)
    # turn off axis if this is called for

    if axis_off:
        ax.axis('off')

    plt.savefig(outname, dpi=dpi)

    # clear_array = [scat_fg, lines_st, p]
    # for i in range(len(clear_array)):
    #     clear_array[i].remove()

    return [scat_fg, lines_st, p]


def data2stills_2Dgyros(datadir, simoutdir, params, framedir_name='stills', init_skip=10, climv=0.1,
                        numbering='adopt', rough=False, roughmov=True, exaggerate=1.0, rm_stills=True,
                        resolution=150, figsize='auto', color_particles='k',
                        DOSexcite=None, lp=None, framerate=10, mov_exten='', title='',
                        dos_meshfn_dir=None, movname=None, lw=None, axis_off=False, **kwargs):
    """Converts a list of data into a stack of png images of gyroscopic lattice using timestep_plot for each timestep.

    Parameters
    ----------
    simoutdir : string
        The output directory for the simulation (contains subdirs for xyv, KL)
    params : dict
        Parameters dictionary
    framedir_name : string
        Subdirectory of simoutdir in which to save movie images
    vsaved : bool
        whether the velocites are recorded (vsaved = False for Nash gyros, True for gHST, for example)
    init_skip : int
        One out of every init_skip frames will be written first, then the intermittent frames will be written, to see
        briefly what happens
    climv : float or tuple
        Color limit for coloring bonds by bond strain
    numbering : 'natural' or 'adopt' (default = 'adopt')
        Use indexing '0','1','2','3',... or adopt the index of the input file.
    rough : boolean
        Plot every init_skip files ONLY? (if False, writes every init_skip files first, then does the rest)
    exaggerate : float (default 1.0 --> in which case it is ignored)
        Exaggerate the displacements of each particle from its initial position by this factor. Ignored if == 1.0
    rm_stills : bool
        Whether or not to delete the stills after making them.
    DOSexcite : tuple of floats or None
        (excitation frequency, stdev time), or else None if DOS plot is not desired
    lp : dict
        Lattice parameters. If not None, then if eigval is not found in main dir, attempts to load eigval from gyro
        network, but will not compute it
    framerate : int or float (optional, default=10)
        framerate for movie
    mov_exten : str (optional)
        additional description to append to movie name, if movname is None
    movname : str or None
        Name or full path with name of movie to output of the simulation
    axis_off : bool
        Turn off axis frame for movie excitation
    **kwargs : keyword arguments for leplt.timestep_plot()
        such as bgcolor, cmap (the strain colormap), color_particles


    Returns
    ----------
    """
    plt.close('all')
    print 'Running data2stills_2Dgyros with DOSexcite = ', DOSexcite
    # get dirs
    # vsaved denotes whether the velocites are recorded
    # vsaved = False for Nash gyros, True for gHST, for example
    print 'simoutdir = ', simoutdir
    try:
        xypath = sorted(glob.glob(simoutdir + 'xyv/'))[0]
        vsaved = True
    except IndexError:
        xypath = sorted(glob.glob(simoutdir + 'xy/'))[0]
        vsaved = False
    # list files
    xyfiles = sorted(glob.glob(xypath + '*.txt'))
    # load setup
    NLfile = sorted(glob.glob(datadir + 'NL.txt'))[0]
    NL = np.loadtxt(NLfile, dtype='int', delimiter=',')
    xy0file = sorted(glob.glob(datadir + 'xy.txt'))[0]
    xy0 = np.loadtxt(xy0file, delimiter=',', usecols=(0, 1))

    if 'deform' in params:
        if params['deform']:
            deform_xy0 = True
            xy0path = sorted(glob.glob(simoutdir + 'xy0/'))[0]
            xy0files = sorted(glob.glob(xy0path + '*.txt'))
        else:
            deform_xy0 = False
            xy0files = []

    try:
        KLpath = sorted(glob.glob(simoutdir + 'KL/'))[0]
        KLfiles = sorted(glob.glob(KLpath + '*.txt'))
        if KLfiles:
            print 'found KLfiles --> update KL each timestep'
            update_KL_each_timestep = True
            KL = np.loadtxt(KLfiles[0], delimiter=',')
        else:
            print 'KLfiles =', KLfiles, '\n --> do not update KL'
            update_KL_each_timestep = False
            KL = np.loadtxt(datadir + 'KL.txt', dtype='int', delimiter=',')
            BM0 = le.NL2BM(xy0, NL, KL)
    except IndexError:
        print 'no KLfiles --> do not update KL'
        update_KL_each_timestep = False
        KL = np.loadtxt(datadir + 'KL.txt', dtype='int', delimiter=',')
        BM0 = le.NL2BM(xy0, NL, KL)

    if 'h' in params:
        hh = params['h']
    elif 'hh' in params:
        hh = params['hh']
    else:
        hfile = sorted(glob.glob(datadir + 'h.txt'))[0]
        hh = np.loadtxt(hfile)

    # get base name from xyfile
    name = 'still'
    if vsaved:
        # name = (xyfiles[0].split('/')[-1]).split('xyv')[0]
        try:
            x, y, vx, vy = np.loadtxt(xyfiles[0], delimiter=',', unpack=True)
        except:
            x, y, z, vx, vy, vz = np.loadtxt(xyfiles[0], delimiter=',', unpack=True)
    else:
        # name = (xyfiles[0].split('/')[-1]).split('xy')[0]
        try:
            '''Data is 2D'''
            x, y = np.loadtxt(xyfiles[0], delimiter=',', unpack=True)
        except:
            try:
                '''Data is 3D'''
                x, y, z = np.loadtxt(xyfiles[0], delimiter=',', unpack=True)
            except:
                '''Data is X,Y,dX,dY'''
                X, Y, dX, dY = np.loadtxt(xyfiles[0], delimiter=',', unpack=True)

    # get length of index string from xyfile
    index_sz = str(len((xyfiles[0].split('_')[-1]).split('.')[0]))
    # make output dir
    outdir = simoutdir + framedir_name + '/'
    dio.ensure_dir(outdir)
    # set range of window from first values
    xlimv = np.ceil(max(x) * 5. / 4.)
    ylimv = np.ceil(max(y) * 5. / 4.)

    # Initial bond list and
    # count initial bonds (double counted)
    # nzcount = np.count_nonzero(KL)
    BL0 = le.NL2BL(NL, KL)
    bo = le.bond_length_list(xy0, BL0)

    # make list of indices to plot-- first sparse then dense
    do1 = [0] + range(0, len(xyfiles), init_skip)

    # Set up figure
    if figsize == 'auto':
        fig = plt.gcf()
        plt.clf()
    else:
        plt.close('all')
        fig = plt.figure(figsize=figsize)

    print 'DOSexcite = ', DOSexcite
    if DOSexcite is not None:
        # Load DOS eigvals:
        eigvalpklglob = glob.glob(datadir + 'eigval.pkl')
        if eigvalpklglob:
            with open(datadir + 'eigval.pkl', "rb") as input_file:
                eigval = cPickle.load(input_file)
            eval_loaded = True
        else:
            print 'Did not find eigval in simulation dir (datadir), attempting to load based on supplied meshfn...'
            # If you want to load eigvals from a lattice other than the one being simulated, put a "pointer file"
            # txt file with the path to that meshfn in your simulation directory: for ex, put 'meshfn_eigvals.txt'
            # in the simdir, with contents '/Users/username/...path.../hexagonal_square_delta0p667_...000010_x_000010/'
            if dos_meshfn_dir is None or dos_meshfn_dir == 'none':
                dos_meshfn_dir = datadir

            meshfn_specfn = glob.glob(dio.prepdir(dos_meshfn_dir) + 'meshfn_eig*.txt')
            print 'dos_meshfn_dir = ', dos_meshfn_dir
            print 'meshfn_specfn = ', meshfn_specfn
            if meshfn_specfn:
                with open(meshfn_specfn[0], 'r') as myfile:
                    meshfn = myfile.read().replace('\n', '')
                if lp is not None:
                    # Build correct eigval to load based on lp (gyro lattice parameters) by grabbing lp[meshfn_exten]
                    import lepm.lattice_class
                    import lepm.gyro_lattice_class
                    # lp = {'LatticeTop': params['LatticeTop'], 'meshfn': params['meshfn']}
                    lat = lepm.lattice_class.Lattice(lp=lp)
                    lat.load()
                    glat = lepm.gyro_lattice_class.GyroLattice(lat, lp)
                    eigvalfn = dio.prepdir(meshfn) + 'eigval' + glat.lp['meshfn_exten'] + '.pkl'
                else:
                    print 'since no lp supplied, assuming eigval is default in datadir...'
                    eigvalfn = dio.prepdir(meshfn) + 'eigval.pkl'
                with open(eigvalfn, "rb") as fn:
                    eigval = cPickle.load(fn)
                eval_loaded = True
            else:
                print 'plotting.time_domain_gyros: Did not find eigval or eigval pointer file in datadir, ' +\
                      'attempting to load based on lp...'
                if lp is not None:
                    print 'Loading based on lp...'
                    # No eigval saved in lattice GyroLattice's meshfn, seeking alternative
                    import lepm.lattice_class
                    import lepm.gyro_lattice_class
                    # lp = {'LatticeTop': params['LatticeTop'], 'meshfn': params['meshfn']}
                    lat = lepm.lattice_class.Lattice(lp=lp)
                    lat.load()
                    print 'lp = ', lp
                    print 'lp[Omk] = ', lp['Omk']
                    glat = lepm.gyro_lattice_class.GyroLattice(lat, lp)
                    eigval = glat.load_eigval()
                    if eigval is None:
                        eigval = glat.get_eigval()
                        print 'Calculated eigval based on supplied GyroLattice instance, using lp dictionary.'
                    else:
                        print 'Loaded eigval from disk, from a location determined by the lp dictionary.'
                    eval_loaded = True
                else:
                    eval_loaded = False
                    raise RuntimeError('Did not supply lp and eigval is not in datadir!')

        if eval_loaded:
            # Attempt to load ipr for network
            iprglob = glob.glob(datadir + 'ipr.pkl')
            if iprglob:
                with open(datadir + 'ipr.pkl', "rb") as input_file:
                    ipr = cPickle.load(input_file)
                    colorV = 1. / ipr
                    linewidth = 0
                    cax_label = r'$p$'
                    colormap = 'viridis_r'
                    vmin_hdr = None
                    vmax_hdr = None
                    cbar_ticklabels = None
                    cbar_nticks = 4
            else:
                locglob = glob.glob(datadir + 'localization*.txt')
                if locglob:
                    localization = np.loadtxt(locglob[0], delimiter=',')
                    ill = localization[:, 2]
                    ill_full = np.zeros(len(eigval), dtype=float)
                    ill_full[0:int(len(eigval) * 0.5)] = ill[::-1]
                    ill_full[int(len(eigval) * 0.5):len(eigval)] = ill
                    colorV = ill_full
                    linewidth = 0
                    cax_label = r'$\lambda^{-1}$'
                    colormap = 'viridis'
                    vmin_hdr = 0.0
                    vmax_hdr = 1. / (np.max(np.abs(xy0.ravel())))
                    cbar_ticklabels = ['0', r'$1/L$', r'$2/L$']
                    cbar_nticks = 3
                else:
                    print 'plotting.time_domain_gyros: Did not find ipr in simulation dir (datadir), attempting ' +\
                          'to load based on supplied meshfn...'
                    # First seek directly supplied files in the simulation datadir
                    meshfn_specfn = glob.glob(datadir + 'meshfn_*ipr.txt')
                    if meshfn_specfn:
                        with open(meshfn_specfn[0], 'r') as myfile:
                            meshfn = myfile.read().replace('\n', '')
                        with open(dio.prepdir(meshfn) + 'ipr' + lp['meshfn_exten'] + '.pkl', "rb") as fn:
                            ipr = cPickle.load(fn)
                        colorV = 1. / ipr
                        cax_label = r'$p$'
                        colormap = 'viridis_r'
                        linewidth = 0
                        vmin_hdr = None
                        vmax_hdr = None
                        cbar_ticklabels = None
                        cbar_nticks = 4
                    else:
                        print '\n\n\nComputing localization from supplied meshfn\n\n\n'
                        if dos_meshfn_dir is None or dos_meshfn_dir == 'none':
                            dos_meshfn_dir = datadir

                        meshfn_specfn = glob.glob(dio.prepdir(dos_meshfn_dir) + 'meshfn_*localization.txt')
                        print 'meshfn_specfn = ', meshfn_specfn
                        if meshfn_specfn:
                            with open(meshfn_specfn[0], 'r') as myfile:
                                meshfn = myfile.read().replace('\n', '')
                            if lp is not None:
                                print 'Loading based on lp...'
                                import lepm.lattice_class
                                import lepm.gyro_lattice_class
                                # lp = {'LatticeTop': params['LatticeTop'], 'meshfn': params['meshfn']}
                                lat = lepm.lattice_class.Lattice(lp=lp)
                                lat.load()
                                glat = lepm.gyro_lattice_class.GyroLattice(lat, lp)
                                loczfn = dio.prepdir(meshfn) + 'localization' + glat.lp['meshfn_exten'] + '.txt'
                                specmeshfn_xy = lat.xy
                            else:
                                print 'plotting.time_domain_gyros: no lp supplied, assuming default in attempt ' +\
                                      'to load localization...'
                                loczfn = dio.prepdir(meshfn) + 'localization.txt'
                                try:
                                    specmeshfn_xy = np.loadtxt(meshfn + '_xy.txt')
                                except:
                                    specmeshfn_xy = np.loadtxt(meshfn + '_xy.txt', delimiter=',')

                            localization = np.loadtxt(loczfn, delimiter=',')
                            ill = localization[:, 2]
                            ill_full = np.zeros(len(eigval), dtype=float)
                            ill_full[0:int(len(eigval) * 0.5)] = ill[::-1]
                            ill_full[int(len(eigval) * 0.5):len(eigval)] = ill
                            colorV = ill_full
                            linewidth = 0
                            cax_label = r'$\lambda^{-1}$'
                            colormap = 'viridis'
                            vmin_hdr = 0.0
                            vmax_hdr = 1. / (np.max(np.abs(specmeshfn_xy.ravel())))
                            cbar_ticklabels = ['0', r'$1/L$', r'$2/L$']
                            cbar_nticks = 3
                        else:
                            print 'plotting.time_domain_gyros: Did not find ipr or localization in datadirs, ' +\
                                  'attempting to load based on lp...'
                            if lp is not None:
                                print 'Loading based on lp...'
                                import lepm.lattice_class
                                import lepm.gyro_lattice_class
                                # lp = {'LatticeTop': params['LatticeTop'], 'meshfn': params['meshfn']}
                                lat = lepm.lattice_class.Lattice(lp=lp)
                                lat.load()
                                glat = lepm.gyro_lattice_class.GyroLattice(lat, lp)
                                if glat.lp['periodicBC']:
                                    localization = glat.get_localization()
                                    ill = localization[:, 2]
                                    ill_full = np.zeros(len(eigval), dtype=float)
                                    ill_full[0:int(len(eigval) * 0.5)] = ill[::-1]
                                    ill_full[int(len(eigval) * 0.5):len(eigval)] = ill
                                    colorV = ill_full
                                    cax_label = r'$\lambda^{-1}$'
                                    vmin_hdr = 0.0
                                    vmax_hdr = 1./(np.max(np.abs(xy0.ravel())))
                                else:
                                    ipr = glat.get_ipr()
                                    colorV = 1. / ipr
                                    cax_label = r'$p$'
                                    colormap = 'viridis_r'
                                    vmin_hdr = None
                                    vmax_hdr = None
                                    cbar_ticklabels = None
                                    cbar_nticks = 4
                                linewidth = 0
                            else:
                                print 'Did not supply lp and neither ipr nor localization are in datadir!'
                                colorV = None
                                linewidth = 1
                                cax_label = ''
                                colormap = 'viridis'
                                vmin_hdr = None
                                vmax_hdr = None
                                cbar_ticklabels = None
                                cbar_nticks = 4

            plt.close('all')
            if np.max(xy0[:, 0]) - np.min(xy0[:, 0]) > 2.0 * (np.max(xy0[:, 1]) - np.min(xy0[:, 1])):
                # Plot will be very wide, so initialize a wide plot (landscape 16:9)
                orientation = 'landscape'
                # Note: header axis is [0.30, 0.80, 0.45, 0.18]
                if title == '' or title is None:
                    ax_pos = [0.1, 0.05, 0.8, 0.54]
                    cbar_pos = [0.79, 0.80, 0.012, 0.15]
                else:
                    ax_pos = [0.1, 0.03, 0.8, 0.50]
                    cbar_pos = [0.79, 0.70, 0.012, 0.15]
            else:
                # Plot will be roughly square or tall, so initialize a portfolio-style plot
                orientation = 'portrait'
                if title == '' or title is None:
                    ax_pos = [0.1, 0.10, 0.8, 0.60]
                    cbar_pos = [0.79, 0.80, 0.012, 0.15]
                else:
                    ax_pos = [0.1, 0.03, 0.8, 0.60]
                    cbar_pos = [0.79, 0.75, 0.012, 0.15]

            # Determine line width
            if lw is None:
                if len(xy0) > 2000:
                    # design lw to be 1 at 15000 particles
                    lw = min(2., np.sqrt(15000.) / np.sqrt(len(xy0)))
                else:
                    lw = 2

            if cax_label == r'$\lambda^{-1}$':
                if 'penrose' in lp['LatticeTop']:
                    dos_ylabel = 'Density of states, ' + r'$D(\omega)$' + '\nfor periodic approximant'
                else:
                    dos_ylabel = 'Density of states, ' + r'$D(\omega)$' + '\nfor periodic system'
                ylabel_pad = 30
                ylabel_rot = 90
            else:
                dos_ylabel = r'$D(\omega)$'
                ylabel_pad = 20
                ylabel_rot = 0

            print 'plt.get_fignums() = ', plt.get_fignums()
            fig, DOS_ax, ax = \
                leplt.initialize_eigvect_DOS_header_plot(eigval, xy0, sim_type='gyro',
                                                         page_orientation=orientation,
                                                         ax_pos=ax_pos, cbar_pos=cbar_pos,
                                                         colorV=colorV, vmin=vmin_hdr, vmax=vmax_hdr,
                                                         DOSexcite=DOSexcite, linewidth=linewidth,
                                                         cax_label=cax_label, colormap=colormap,
                                                         cbar_nticks=cbar_nticks,
                                                         cbar_tickfmt='%0.2f', cbar_ticklabels=cbar_ticklabels,
                                                         cbar_labelpad=17,
                                                         yaxis_ticks=[], ylabel=dos_ylabel,
                                                         ylabel_rot=ylabel_rot, ylabel_pad=ylabel_pad,
                                                         nbins=120, xlabel_pad=15)
            # DOSexcite = (frequency, sigma_time)
            # amp(x) = exp[- acoeff * time**2]
            # amp(k) = sqrt(pi/acoeff) * exp[- pi**2 * k**2 / acoeff]
            # So 1/(2 * sigma_freq**2) = pi**2 /acoeff
            # So sqrt(acoeff/(2 * pi**2)) = sigma_freq

            # sigmak = 1./DOSexcite[1]
            # xlims = DOS_ax.get_xlim()
            # ktmp = np.linspace(xlims[0], xlims[1], 300)
            # gaussk = 0.8 * DOS_ax.get_ylim()[1] * np.exp(-(ktmp - DOSexcite[0])**2 / (2. * sigmak))
            # DOS_ax.plot(ktmp, gaussk, 'r-')
            # plt.sca(ax)
        else:
            print 'Could not find eigval.pkl to load for DOS portion of data2stills plots!'
            ax = plt.gca()
    else:
        ax = plt.gca()

    # Check for evolving rest lengths in params
    if 'prestrain' in params:
        prestrain = params['prestrain']
    else:
        prestrain = 0.

    if 'shrinkrate' in params:
        shrinkrate = params['shrinkrate']
    else:
        shrinkrate = 0.0

    if roughmov:
        print 'creating rough gyro movie...'
        stills2mov_gyro(fig, ax, do1, xyfiles, KLfiles, xy0files, xy0, NL, KL, BM0, params, hh,
                        numbering, index_sz,
                        outdir, name, simoutdir, update_KL_each_timestep, deform_xy0, exaggerate, xlimv,
                        ylimv, climv,
                        resolution, color_particles, shrinkrate, prestrain, framerate=float(framerate) / 5.,
                        mov_exten='_rough', linewidth=lw, startind=0, title=title, axis_off=axis_off, **kwargs)

    # Now do detailed movie if rough is False
    if not rough:
        print 'creating fine gyro movie (not skipping any frames)...'
        doall = [0] + range(0, len(xyfiles))
        # do2 = list(set(doall)-set(do1))
        # ftodo = do1 + do2

        stills2mov_gyro(fig, ax, doall, xyfiles, KLfiles, xy0files, xy0, NL, KL, BM0, params, hh,
                        numbering, index_sz,
                        outdir, name, simoutdir, update_KL_each_timestep, deform_xy0, exaggerate, xlimv,
                        ylimv, climv,
                        resolution, color_particles, shrinkrate, prestrain, framerate=framerate,
                        mov_exten=mov_exten,
                        linewidth=lw, startind=0, title=title, movname=movname, axis_off=axis_off, **kwargs)

        if rm_stills:
            # Delete the original images
            print 'Deleting folder ' + simoutdir + 'stills/'
            subprocess.call(['rm', '-r', simoutdir + 'stills/'])


def data2stills_plot_deforming_xy0(datadir, params, framedir_name='stills_ref', init_skip=10, climv=0.1,
                                   numbering='adopt', rough=False, rm_stills=True, resolution='auto', figsize='auto',
                                   **kwargs):
    """Converts a list of reference lattice data (xy0) into a stack of png images of deforming lattice using
    for each timestep.

    Parameters
    ----------
    datadir : string
        The output directory for the simulation (contains subdirs for xyv, KL)
    params : dict
        Parameters dictionary
    framedir_name : string
        Subdirectory of datadir in which to save movie images
    init_skip : int
        One out of every init_skip frames will be written first, then the intermittent frames will be written,
        to see briefly what happens
    climv : float or tuple
        Color limit for coloring bonds by bond strain
    numbering : 'natural' or 'adopt' (default = 'adopt')
        Use indexing '0','1','2','3',... or adopt the index of the input file.
    rough : boolean
        Plot every init_skip files ONLY? (if False, writes every init_skip files first, then does the rest)
    rm_stills : bool
        Whether or not to delete the stills after making them.
    **kwargs : keyword arguments for leplt.le_plot_lattice()

    Returns
    ----------
    """
    # get dirs
    # vsaved denotes whether the velocites are recorded
    # vsaved = False for Nash gyros, True for gHST, for example
    xypath = sorted(glob.glob(datadir + 'xy0/'))[0]
    KLpath = sorted(glob.glob(datadir + 'KL/'))[0]
    # list files
    xyfiles = sorted(glob.glob(xypath + '*.txt'))
    KLfiles = sorted(glob.glob(KLpath + '*.txt'))
    # load setup
    NLfile = sorted(glob.glob(datadir + 'NL.txt'))[0]
    NL = np.loadtxt(NLfile, dtype='int', delimiter=',')
    xy0file = sorted(glob.glob(datadir + 'xy.txt'))[0]
    xy0_orig = np.loadtxt(xy0file, delimiter=',', usecols=(0, 1))

    xy0path = sorted(glob.glob(datadir + 'xy0/'))[0]
    xy0files = sorted(glob.glob(xy0path + '*.txt'))

    if KLfiles:
        print 'KLfiles =', KLfiles, '\n --> update KL each timestep'
        update_KL_each_timestep = True
        KL = np.loadtxt(KLfiles[0], delimiter=',')
    else:
        print 'KLfiles =', KLfiles, '\n --> do not update KL'
        update_KL_each_timestep = False
        KL = np.loadtxt(datadir + 'KL.txt', dtype='int', delimiter=',')
        BM0 = le.NL2BM(xy0_orig, NL, KL)

    try:
        hh = params['h']
    except:
        hfile = sorted(glob.glob(datadir + 'h.txt'))[0]
        hh = np.loadtxt(hfile)

    # get base name from xyfile
    name = 'still'

    # get length of index string from xyfile
    index_sz = str(len((xyfiles[0].split('_')[-1]).split('.')[0]))
    # make output dir
    outdir = datadir + framedir_name + '/'
    dio.ensure_dir(outdir)
    # set range of window from first values
    xlimv = np.ceil(max(xy0_orig[:, 0]) * 5. / 4.)
    ylimv = np.ceil(max(xy0_orig[:, 1]) * 5. / 4.)

    # Initial bond list and
    # count initial bonds (double counted)
    # nzcount = np.count_nonzero(KL)
    BL0 = le.NL2BL(NL, KL)
    bo = le.bond_length_list(xy0_orig, BL0)

    # make list of indices to plot-- first sparse then dense
    do1 = range(0, len(xyfiles), init_skip)
    if not rough:
        doall = range(0, len(xyfiles))
        do2 = list(set(doall) - set(do1))
        doall.append(0)
        ftodo = do1 + do2
    else:
        ftodo = do1

    # Set up figure
    if figsize == 'auto':
        fig = plt.gcf()
        plt.clf()
    else:
        plt.close('all')
        fig = plt.figure(figsize=figsize)

    ax = plt.gca()

    # Get colorbar position by doing plot once
    ax.cla()
    axcb = 'none'

    # Go through xy0 data and save as stills
    for i in ftodo:
        print('Saving still:' + str(i) + '/' + str(len(ftodo)))

        # update xy0 as lattice is deforming
        xy0 = np.loadtxt(xy0files[i], delimiter=',', usecols=(0, 1))
        iterind = (xyfiles[i].split('_')[-1]).split('.')[0]

        if numbering == 'natural':
            index = ('{0:0' + index_sz + 'd}').format(i)
        else:
            index = iterind

        # get timestamp for this image
        t = float(iterind) * hh

        # Recalculate BM if KL is evolving
        if update_KL_each_timestep:
            KL = np.loadtxt(KLfiles[i], delimiter=',')
            # Bond matrix: connectivity of lattice AND rest length info
            # BM0 = NL2BM(xy0, NL,KL)

        [ax, axcb] = le_plot_lattice(xy0, xy0_orig, NL, KL, BL0, bo, params, t, i, name, outdir,
                                     climv=climv, exaggerate=1.0, colorz=False, ax=ax, axcb=axcb, **kwargs)
        ax.cla()

    # MAKE MOVIE
    hostdir = outdir
    fname, index_sz = dio.get_fname_and_index_size(hostdir)
    imgname = hostdir + fname
    paths = datadir.split('/')
    datedir = ''
    for ii in range(len(paths) - 2):
        pn = paths[ii]
        datedir += pn + '/'
    movname = datedir + datadir.split('/')[-2] + '_xy0'

    subprocess.call(['./ffmpeg', '-framerate', str(framerate), '-i', imgname + '%' + str(index_sz) + 'd.png',
                     movname + '.mov', '-vcodec', 'libx264',
                     '-profile:v', 'main', '-crf', '12', '-threads', '0', '-r', '100', '-pix_fmt', 'yuv420p'])

    if rm_stills:
        # Delete the original images
        print 'Deleting folder ' + datadir + framedir_name + '/'
        subprocess.call(['rm', '-r', datadir + framedir_name + '/'])


def stills2mov_gyro(fig, ax, ftodo, xyfiles, KLfiles, xy0files, xy0, NL, KL, BM0, params, hh, numbering, index_sz,
                    outdir, name, datadir, update_KL_each_timestep, deform_xy0, exaggerate, xlimv, ylimv, climv,
                    resolution, color_particles, shrinkrate, prestrain, framerate=10, mov_exten='', linewidth=2,
                    startind=0, title='', movname=None, **kwargs):
    """Make a sequence of images for a gyro network and save it as a movie.
    This is a utility function called by data2stills_2Dgyros() to create stills and write movie.

    Parameters
    ----------
    todo : int array or list
        The indices of the stills to plot and save in a movie
    xyfiles : list of strings
        paths of xy data to load
    KLfiles : list of strings
        paths of connectivity files to load, if connectivity changes during sequence
    xy0files : list of strings or empty list if xy0 is not evolving
    xy0 : NP x 2 float array
        pinning sites of the gyros
    BM0 : NP x max #NN float array
        initial bond matrix
    params : dict
        parameters for the simulation
    hh : float
        timestep
    numbering : string specifier ('natural' or other)
        Whether to place the index of the frame as the index of the output timestep or of the frame index
    index_sz : int
        How many digits in the index (with leading zeros)
    outdir : str
        path for the frames
    name : str
        name of still frame before index
    datadir : str
        path for the movie
    update_KL_each_timestep : bool
        Whether connectivity is changing during simulation
    deform_xy0 : bool
        Whether rest positions for gyros are moving during sim
    exaggerate : float
        Exaggeration of the displacement
    xlimv : tuple of floats
        limits for x axis
    ylimv : tuple of floats
        limits for y axis
    climv : tuple of floats
        limits for color axis
    mov_exten : str (optional, default is '')
        an extension to the movie name describing the movie
    **kwargs : keyword arguments for lepm.plotting.plotting.le_plot_gyros()
        axis_off=False
    """
    # Go through data and save as stills
    first = True
    natind = startind
    for i in ftodo:
        print 'Saving still:' + str(i) + '/' + str(len(ftodo))

        xy = np.loadtxt(xyfiles[i], delimiter=',', usecols=(0, 1))
        iterind = (xyfiles[i].split('_')[-1]).split('.')[0]

        # if numbering == 'natural':
        #     index = ('{0:0' + index_sz + 'd}').format(i)
        # else:
        #     index = iterind

        # get timestamp for this image
        tt = float(iterind) * hh

        # Recalculate BM if KL is evolving
        if update_KL_each_timestep:
            KL = np.loadtxt(KLfiles[i], delimiter=',')
            # Bond matrix: connectivity of lattice AND rest length info
            BM0 = le.NL2BM(xy0, NL, KL)

        # calculate strain
        BM = BM0 * (1. - shrinkrate * tt - prestrain)

        # update xy0 if lattice is deforming
        if deform_xy0:
            xy0 = np.loadtxt(xy0files[i], delimiter=',', usecols=(0, 1))

        if not first:
            title = None

        # print 'tdgyros: color_particles = ', color_particles
        # sys.exit()
        [scat_fg, lines_st, p] = le_plot_gyros(xy, xy0, NL, KL, BM, params, tt, natind, name, fig, ax, outdir,
                                               climv=climv, exaggerate=exaggerate, dpi=resolution,
                                               color_particles=color_particles,
                                               linewidth=linewidth, title=title, **kwargs)
        ax.cla()
        # Ignore the first plot since the size of the gyro markers will be wrong
        if first:
            first = False
        else:
            natind += 1

    # MAKE MOVIE
    hostdir = datadir + 'stills/'
    fname, index_sz = dio.get_fname_and_index_size(hostdir)
    imgname = hostdir + fname
    paths = datadir.split('/')
    if movname is None:
        datedir = ''
        for ii in range(len(paths) - 2):
            pn = paths[ii]
            datedir += pn + '/'
        movname = datedir + datadir.split('/')[-2] + '_gyro' + mov_exten
    elif '/' not in movname:
        # If the supplied movname is not a full path, place it in the datedir
        movname = datadir + movname

    print datadir
    subprocess.call(['./ffmpeg', '-framerate', str(framerate), '-i', imgname + '%' + str(index_sz) + 'd.png',
                     movname + '.mov', '-vcodec', 'libx264', '-profile:v', 'main', '-crf', '12',
                     '-threads', '0', '-r', '30', '-pix_fmt', 'yuv420p'])


