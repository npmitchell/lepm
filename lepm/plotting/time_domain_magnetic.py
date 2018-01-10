import subprocess
import matplotlib.pyplot as plt
import glob
import numpy as np
import cPickle
import lepm.dataio as dio
import lepm.plotting.plotting as leplt
import lepm.lattice_elasticity as le
import lepm.plotting.time_domain_gyros as tdgyros

"""Functions for plotting magnetic networks in time domain simulations
"""


def data2stills_2Dgyros(datadir, simoutdir, params, framedir_name='stills', init_skip=10, climv=0.1,
                        numbering='adopt', rough=False, roughmov=True, exaggerate=1.0, rm_stills=True,
                        resolution=150, figsize='auto', color_particles='k',
                        DOSexcite=None, lp=None, framerate=10, mov_exten='', title='',
                        dos_meshfn_dir=None, movname=None, lw=2, **kwargs):
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
    **kwargs : keyword arguments for leplt.timestep_plot()
        such as bgcolor, cmap (the strain colormap), color_particles, mimic_expt


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
                    mlat = lepm.magnetic_gyro_lattice_class.MagneticGyroLattice(lat, lp)
                    eigvalfn = dio.prepdir(meshfn) + 'eigval' + mlat.lp['meshfn_exten'] + '.pkl'
                else:
                    print 'since no lp supplied, assuming eigval is default in datadir...'
                    eigvalfn = dio.prepdir(meshfn) + 'eigval_magnetic.pkl'
                with open(eigvalfn, "rb") as fn:
                    eigval = cPickle.load(fn)
                eval_loaded = True
            else:
                print 'plotting.time_domain_magnetic: Did not find eigval or eigval pointer file in datadir, ' + \
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
                    mlat = lepm.magnetic_gyro_lattice_class.MagneticGyroLattice(lat, lp)
                    eigval = mlat.load_eigval()
                    if eigval is None:
                        eigval = mlat.get_eigval()
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
                    print 'plotting.time_domain_magnetic: Did not find ipr in simulation dir (datadir), ' +\
                          'attempting to load based on supplied meshfn...'
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
                                mlat = lepm.magnetic_gyro_lattice_class.MagneticGyroLattice(lat, lp)
                                loczfn = dio.prepdir(meshfn) + 'localization' + mlat.lp['meshfn_exten'] + '.txt'
                                specmeshfn_xy = lat.xy
                            else:
                                print 'plotting.time_domain_magnetic: no lp supplied, assuming default lattice ' +\
                                      'params to load localization in attempt to load localization...'
                                loczfn = dio.prepdir(meshfn) + 'localization_magnetic.txt'
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
                            print 'plotting.time_domain_magnetic: Did not find ipr or localization in datadirs,' +\
                                  ' attempting to load based on lp...'
                            if lp is not None:
                                print 'Loading based on lp...'
                                import lepm.lattice_class
                                import lepm.gyro_lattice_class
                                # lp = {'LatticeTop': params['LatticeTop'], 'meshfn': params['meshfn']}
                                lat = lepm.lattice_class.Lattice(lp=lp)
                                lat.load()
                                mlat = lepm.magnetic_gyro_lattice_class.MagneticGyroLattice(lat, lp)
                                if mlat.lp['periodicBC']:
                                    localization = mlat.get_localization()
                                    ill = localization[:, 2]
                                    ill_full = np.zeros(len(eigval), dtype=float)
                                    ill_full[0:int(len(eigval) * 0.5)] = ill[::-1]
                                    ill_full[int(len(eigval) * 0.5):len(eigval)] = ill
                                    colorV = ill_full
                                    cax_label = r'$\lambda^{-1}$'
                                    vmin_hdr = 0.0
                                    vmax_hdr = 1./(np.max(np.abs(xy0.ravel())))
                                else:
                                    ipr = mlat.get_ipr()
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
            if len(xy0) > 2000:
                lw = 1
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
        tdgyros.stills2mov_gyro(fig, ax, do1, xyfiles, KLfiles, xy0files, xy0, NL, KL, BM0, params, hh,
                                numbering, index_sz,
                                outdir, name, simoutdir, update_KL_each_timestep, deform_xy0, exaggerate, xlimv,
                                ylimv, climv,
                                resolution, color_particles, shrinkrate, prestrain, framerate=float(framerate) / 5.,
                                mov_exten='_rough', linewidth=lw, startind=0, title=title, show_bonds=False, **kwargs)

    # Now do detailed movie if rough is False
    if not rough:
        print 'creating fine gyro movie (not skipping any frames)...'
        doall = [0] + range(0, len(xyfiles))
        # do2 = list(set(doall)-set(do1))
        # ftodo = do1 + do2

        print 'tdmagnetic: exiting here since it is broken here'
        # sys.exit()
        tdgyros.stills2mov_gyro(fig, ax, doall, xyfiles, KLfiles, xy0files, xy0, NL, KL, BM0, params, hh,
                                numbering, index_sz,
                                outdir, name, simoutdir, update_KL_each_timestep, deform_xy0, exaggerate, xlimv,
                                ylimv, climv,
                                resolution, color_particles, shrinkrate, prestrain, framerate=framerate,
                                mov_exten=mov_exten,
                                linewidth=lw, startind=0, title=title, movname=movname, show_bonds=False, **kwargs)

        if rm_stills:
            # Delete the original images
            print 'Deleting folder ' + simoutdir + 'stills/'
            subprocess.call(['rm', '-r', simoutdir + 'stills/'])
