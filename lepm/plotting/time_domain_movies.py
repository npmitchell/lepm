import subprocess
import numpy as np
import lepm.lattice_elasticity as le
import lepm.dataio as dio
import glob

'''Lower level function than time_domain_gyros, time_domain_ghst, etc'''


def stills2mov_util(todo, xyfiles, KLfiles, params, h, numbering, index_sz, framedir_name, name,
                    update_KL_each_timestep, exaggerate, xlimv, ylimv, climv, framerate=10, mov_exten=''):
    """Make a sequence of images and save it as a movie. This funciton is a utility function called by data2stills_2D()

    Parameters
    ----------
    todo : int array or list
        The indices of the stills to plot and save in a movie
    xyfiles : list of strings
        paths of xy data to load
    KLfiles : list of strings
        paths of connectivity files to load, if connectivity changes during sequence
    params : dict
        parameters for the simulation
    h : float
        timestep
    numbering : string specifier ('natural' or other)
        Whether to place the index of the frame as the index of the output timestep or of the frame index
    index_sz : int
        How many digits in the index (with leading zeros)
    framedir_name : str
        path for the frames
    name : str
        non-index part of the frame names
    update_KL_each_timestep : bool
        Whether connectivity is changing during simulation
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
    """
    # Go through data and save as stills
    for i in todo:
        if np.mod(i, 50) == 0:
            print 'Saving still:' + str(i) + '/' + str(len(todo))

        xy = np.loadtxt(xyfiles[i], delimiter=',', usecols=(0, 1))

        iterind = (xyfiles[i].split('_')[-1]).split('.')[0]

        if numbering == 'natural':
            index = ('{0:0' + index_sz + 'd}').format(i)
        else:
            index = iterind

        outname = datadir + framedir_name + '/' + name + index + '.png'

        # Recalculate bo for each timestep if KL is evolving
        if update_KL_each_timestep:
            KL = np.loadtxt(KLfiles[i], delimiter=',')

            BL = le.NL2BL(NL, KL)

            # Calc rest bond lengths
            # --> have to do this if bonds are cut so that BL and bo are same size
            # --> can't do if statement if rough is False
            # if np.count_nonzero(KL)!=nzcount: #can only do this is rough is true (or if in order)
            bo = le.bond_length_list(xy0, BL)

        if 'prestrain' in params:
            prestrain = params['prestrain']
        else:
            prestrain = 0.

        if 'shrinkrate' in params:
            shrinkrate = params['shrinkrate']
            title = 't = ' + '%09.4f' % (float(iterind) * h) + '  a = ' + \
                    '%07.5f' % (1. - shrinkrate * float(iterind) * h - prestrain)
        elif 'prestrain' in params:
            shrinkrate = 0.0
            title = 't = ' + '%09.4f' % (float(iterind) * h) + '  a = ' + '%07.5f' % (1. - prestrain)
        else:
            shrinkrate = 0.0
            title = 't = ' + '%09.4f' % (float(iterind) * h)

        print 'Calculating bond strain... '
        # calculate strain
        bs = le.bond_strain_list(xy, BL, bo * (1. - shrinkrate * (float(iterind) * h) - prestrain))
        # if bond lengths are all unity, then simple:
        # bs = bond_strain_list(xy,BL, np.ones_like(BL[:,0]))

        if exaggerate == 1.0:
            movie_plot_2D(xy, BL, bs, outname, title, xlimv=xlimv, ylimv=ylimv, climv=climv)
        else:
            print 'making frame in movie plot'
            xye = xy0 + (xy - xy0) * exaggerate
            movie_plot_2D(xye, BL, bs, outname, title, xlimv=xlimv, ylimv=ylimv, climv=climv)

    # MAKE MOVIE
    hostdir = datadir + 'stills/'
    fname, index_sz = dio.get_fname_and_index_size(hostdir)
    print 'index_sz = ', index_sz
    print 'hostdir = ', hostdir
    fps = 15
    imgname = hostdir + fname
    paths = datadir.split('/')
    datedir = ''
    for ii in range(len(paths) - 2):
        pn = paths[ii]
        datedir += pn + '/'
    movname = datedir + datadir.split('/')[-2]

    subprocess.call(['./ffmpeg', '-framerate', str(framerate), '-i', imgname + '%' + str(index_sz) + 'd.png',
                     movname + mov_exten + '.mov', '-vcodec',
                     'libx264', '-profile:v', 'main', '-crf', '12', '-threads', '0', '-r', '30',
                     '-pix_fmt', 'yuv420p'])