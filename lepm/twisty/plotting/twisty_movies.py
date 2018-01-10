import matplotlib.pyplot as plt
import subprocess
import numpy as np
import lepm.dataio as dio
import lepm.stringformat as sf
import lepm.plotting.plotting as leplt
import lepm.twisty.plotting.plotting as twistyplt
import glob

'''Functions for making movies of TwistyLattice class instances
'''


def save_normal_modes_twisty(tlat, datadir='auto', sim_type='xywz', rm_images=True,
                             save_into_subdir=False, overwrite=True, color='pr', do_bc=False):
    """
    Plot the normal modes of a coupled gyros with fixed pivot points and make a movie of them.

    Parameters
    ----------
    tlat : TwistyLattice instance
        with attributes lp, Omg, xy_inner
    datadir: string
        directory where simulation data is stored
    rm_images : bool
        Whether or not to delete all the images after a movie has been made of the DOS
    save_into_subdir: bool
        Whether to save the movie in the same sudir where the DOS/ directory (containing the frames) is placed.
    """
    NP = len(tlat.xy_inner)
    print 'Getting eigenvals/vects of dynamical matrix...'
    # Find eigval/vect
    eigval, eigvect = tlat.get_eigval_eigvect()

    # prepare DOS output dir
    if datadir == 'auto':
        datadir = tlat.lp['meshfn']

    if 'meshfn_exten' in tlat.lp:
        exten = tlat.lp['meshfn_exten']
    else:
        raise RuntimeError('No meshfn_exten in tlat.lp')

    DOSdir = datadir + 'DOS' + exten + '/'
    dio.ensure_dir(DOSdir)

    #####################################
    # Prepare for plotting
    #####################################
    print 'Preparing plot settings...'

    # Options for how to color the DOS header
    if color == 'pr' or len(tlat.lattice.xy) < 3:
        ipr = tlat.get_ipr()
        colorV = 1./ipr
        lw = 0
        cbar_label = r'$p$'
        colormap = 'viridis_r'
    elif color == 'ipr':
        ipr = tlat.get_ipr()
        colorV = ipr
        lw = 0
        cbar_label = r'$p^{-1}$'
        colormap = 'viridis'
    elif color == 'ill':
        ill = tlat.get_ill()
        colorV = ill
        cbar_label = r'$\lambda^{-1}$'
        lw = 0
        colormap = 'viridis'
    else:
        colorV = None
        lw = 1.
        cbar_label = ''
        colormap = 'viridis'

    print 'plotting...'
    fig, DOS_ax, eig_ax = leplt.initialize_eigvect_DOS_header_plot(eigval, tlat.lattice.xy, sim_type=sim_type,
                                                                   colorV=colorV, colormap=colormap, linewidth=lw,
                                                                   cax_label=cbar_label, cbar_nticks=2,
                                                                   cbar_tickfmt='%0.3f')

    dostitle = r'$k = $' + '{0:0.2f}'.format(tlat.lp['kk'])
    if tlat.lp['ABDelta_k']:
        dostitle += r'$\pm$' + '{0:0.2f}'.format(tlat.lp['ABDelta_k'])
    dostitle += r' $g = $' + '{0:0.2f}'.format(tlat.lp['gg'])
    if tlat.lp['ABDelta_g']:
        dostitle += r'$\pm$' + '{0:0.2f}'.format(tlat.lp['ABDelta_g'])
    dostitle += r' $c = $' + '{0:0.2f}'.format(tlat.lp['cc'])
    if tlat.lp['ABDelta_c']:
        dostitle += r'$\pm$' + '{0:0.2f}'.format(tlat.lp['ABDelta_c'])
    DOS_ax.set_title(dostitle)
    tlat.lattice.plot_BW_lat(fig=fig, ax=eig_ax, save=False, close=False, axis_off=False, title='')

    # Make strings for spring, pin, k, and g values
    text2show = tlat.lp['meshfn_exten'][1:]
    fig.text(0.01, 0.01, text2show, horizontalalignment='left', verticalalignment='bottom')

    #####################################
    # SAVE eigenvals/vects as images
    #####################################
    # First check that we actually need to make the images: if rm_images == True and the movie exists, then stop if
    # overwrite==False

    # Construct movname
    names = DOSdir.split('/')[0:-1]
    # Construct movie name from datadir path string
    movname = ''
    for ii in range(len(names)):
        if ii < len(names)-1:
            movname += names[ii]+'/'
        else:
            if save_into_subdir:
                movname += names[ii] + '/' + names[ii] + '_DOS'
            else:
                movname += names[ii]
                # movname += '_DOS' + gyro_lattice.lp['meshfn_exten']

    if not (not overwrite and rm_images and glob.glob(movname + '.mov')):
        done_pngs = len(glob.glob(DOSdir + 'DOS_*.png'))
        # check if normal modes have already been done
        if not done_pngs:
            # decide on which eigs to plot
            todo = np.arange(len(eigval))
            todo_subset = np.setdiff1d(todo, np.arange(done_pngs))
            todo_subset = np.hstack((np.array([todo_subset[0]]), todo_subset))
            # print 'todo = ', todo_subset
            # sys.exit()

            if done_pngs < len(todo):
                dmyi = 0
                for ii in todo_subset:
                    if np.mod(ii, 50) == 0:
                        print 'plotting eigvect ', ii, ' of ', len(eigval)
                    # tlat.lattice.plot_BW_lat(fig=fig, ax=DOS_ax, save=False, close=False, axis_off=False, title='')
                    fig, [scat_fg, scat_fg2, f_mark, polygons, tiltgons], cw_ccw = \
                        twistyplt.construct_eigvect_DOS_plot(tlat.xy_inner, fig, DOS_ax, eig_ax, eigval, eigvect,
                                                             ii, sim_type, tlat.NL_t, tlat.KL_t,
                                                             marker_num=0, color_scheme='default', sub_lattice=-1)
                    plt.savefig(DOSdir + 'DOS_' + '{0:05}'.format(ii) + '.png')
                    scat_fg.remove()
                    scat_fg2.remove()
                    polygons.remove()
                    tiltgons.remove()
                    # for pp in polygons:
                    #     pp.remove()
                    # for tt in tiltgons:
                    #     tt.remove()

                    f_mark.remove()
                    # plt.show()
                    # sys.exit()
                    # lines_12_st.remove()
                    # eig_ax.cla()
                    if ii > 0:
                        dmyi += 1

        fig.clf()
        plt.close('all')

    print 'eigvect = ', eigvect

    ######################
    # Save DOS as movie
    ######################
    imgname = DOSdir + 'DOS_'
    subprocess.call(['./ffmpeg', '-i', imgname+'%05d.png', movname+'.mov', '-vcodec', 'libx264', '-profile:v', 'main',
                     '-crf', '12', '-threads', '0', '-r', '100', '-pix_fmt', 'yuv420p'])

    if rm_images:
        # Delete the original images
        if not save_into_subdir:
            print 'Deleting folder ' + DOSdir
            subprocess.call(['rm', '-r', DOSdir])
        else:
            print 'Deleting folder contents ' + DOSdir + 'DOS_*.png'
            subprocess.call(['rm', '-r', DOSdir + 'DOS_*.png'])
