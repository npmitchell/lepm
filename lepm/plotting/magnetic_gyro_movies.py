import matplotlib.pyplot as plt
import subprocess
import numpy as np
import lepm.dataio as dio
import lepm.stringformat as sf
import plotting as leplt
import glob

'''Functions for making movies of MagneticGyroLattice class instances
'''


def save_normal_modes_maggyro(mlat, datadir='auto', sim_type='gyro', rm_images=True,
                              gapims_only=False, save_into_subdir=False, overwrite=True, color='pr', do_bc=False):
    """
    Plot the normal modes of a coupled gyros with fixed pivot points and make a movie of them.
    Note that b and c are built INTO the signs of OmK and Omg.
    --> b (0 -> 'hang',  1 -> 'stand').
    --> c (0 -> aligned with a,  1 -> aligned with a)

    Parameters
    ----------
    mlat : MagneticGyroLattice instance
        with attributes lp, Omg, xy_inner
    datadir: string
        directory where simulation data is stored
    rm_images : bool
        Whether or not to delete all the images after a movie has been made of the DOS
    gapims_only : bool
        Whether to just plot modes near the middle of the DOS frequency range
    save_into_subdir: bool
        Whether to save the movie in the same sudir where the DOS/ directory (containing the frames) is placed.
    """
    NP = len(mlat.xy_inner)
    print 'Getting eigenvals/vects of dynamical matrix...'
    # Find eigval/vect
    eigval, eigvect = mlat.get_eigval_eigvect()
    OmK = mlat.lp['Omk'] * mlat.KL_nm
    Omg = mlat.Omg

    # prepare DOS output dir
    if datadir == 'auto':
        datadir = mlat.lp['meshfn']

    if 'meshfn_exten' in mlat.lp:
        exten = mlat.lp['meshfn_exten']
    elif mlat.lp['dcdisorder']:
        # there should be meshfn_exten defined in lp, but since there is not, we make it here
        exten = '_magnetic_pinV' + sf.float2pstr(mlat.lp['V0_pin_gauss']) + \
                '_strV' + sf.float2pstr(mlat.lp['V0_spring_gauss'])
    else:
        exten = '_magnetic'

    DOSdir = datadir + 'DOS' + exten + '/'
    dio.ensure_dir(DOSdir)

    #####################################
    # Prepare for plotting
    #####################################
    print 'Preparing plot settings...'

    # Options for how to color the DOS header
    if color == 'pr' or len(mlat.lattice.xy) < 3:
        ipr = mlat.get_ipr()
        colorV = 1./ipr
        lw = 0
        cbar_label = r'$p$'
        colormap = 'viridis_r'
    elif color == 'ipr':
        ipr = mlat.get_ipr()
        colorV = ipr
        lw = 0
        cbar_label = r'$p^{-1}$'
        colormap = 'viridis'
    elif color == 'ill':
        ill = mlat.get_ill()
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
    fig, DOS_ax, eig_ax = leplt.initialize_eigvect_DOS_header_plot(eigval, mlat.lattice.xy, sim_type=sim_type,
                                                                   colorV=colorV, colormap=colormap, linewidth=lw,
                                                                   cax_label=cbar_label, cbar_nticks=2,
                                                                   cbar_tickfmt='%0.3f')

    dostitle = r'$\Omega_g = $' + '{0:0.2f}'.format(mlat.lp['Omg'])
    if mlat.lp['ABDelta']:
        dostitle += r'$\pm$' + '{0:0.2f}'.format(mlat.lp['ABDelta'])
    dostitle += r' $\Omega_k=$' + '{0:0.2f}'.format(mlat.lp['Omk'])
    DOS_ax.set_title(dostitle)
    mlat.lattice.plot_BW_lat(fig=fig, ax=eig_ax, save=False, close=False, axis_off=False, title='')

    # Make strings for spring, pin, k, and g values
    text2show = mlat.lp['meshfn_exten'][1:]
    fig.text(0.4, 0.1, text2show, horizontalalignment='center', verticalalignment='center')

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

    if not (not overwrite and rm_images and glob.glob(movname+'.mov')):
        done_pngs = len(glob.glob(DOSdir + 'DOS_*.png'))
        # check if normal modes have already been done
        if not done_pngs:
            # decide on which eigs to plot
            totN = len(eigval)
            if gapims_only:
                middle = int(round(totN*0.25))
                ngap = int(round(np.sqrt(totN)))
                todo = range(middle - ngap, middle + ngap)
            else:
                todo = range(int(round(len(eigval)*0.5)))

            if done_pngs < len(todo):
                dmyi = 0
                for ii in todo:
                    if np.mod(ii, 50) == 0:
                        print 'plotting eigvect ', ii, ' of ', len(eigval)
                    # mlat.lattice.plot_BW_lat(fig=fig, ax=DOS_ax, save=False, close=False, axis_off=False, title='')
                    fig, [scat_fg, scat_fg2, p, f_mark, lines_12_st], cw_ccw = \
                        leplt.construct_eigvect_DOS_plot(mlat.xy_inner, fig, DOS_ax, eig_ax, eigval, eigvect,
                                                         ii, sim_type, mlat.NL_nm, mlat.KL_nm,
                                                         marker_num=0, color_scheme='default', sub_lattice=-1)
                    plt.savefig(DOSdir + 'DOS_' + '{0:05}'.format(dmyi) + '.png')
                    scat_fg.remove()
                    scat_fg2.remove()
                    p.remove()
                    f_mark.remove()
                    lines_12_st.remove()
                    dmyi += 1

        fig.clf()
        plt.close('all')

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
