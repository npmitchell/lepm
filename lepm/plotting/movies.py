import matplotlib.pyplot as plt
import subprocess
import numpy as np
import lepm.dataio as dio
import lepm.stringformat as sf
import plotting as leplt
import glob
import lepm.plotting.haldane_lattice_plotting_functions as hlatpfns


def save_normal_modes_Nashgyro(gyro_lattice, datadir='auto', dispersion=[], sim_type='gyro', rm_images=True,
                               gapims_only=False, save_into_subdir=False, overwrite=True, color='pr', do_bc=False):
    """
    Plot the normal modes of a coupled gyros with fixed pivot points and make a movie of them.
    Note that b and c are built INTO the signs of OmK and Omg.
    --> b (0 -> 'hang',  1 -> 'stand').
    --> c (0 -> aligned with a,  1 -> aligned with a)

    Parameters
    ----------
    datadir: string
        directory where simulation data is stored
    R : NP x dim array
        position array in 2d (3d might work with 3rd dim ignored)
    NL : NP x NN array
        Neighbor list
    KL : NP x NN array
        spring connectivity array
    OmK : float or NP x NN array
        OmK (spring frequency array, for Nash limit: (-1)^(c+b)kl^2/Iw'
    Omg : float or NP x 1 array
        gravitational frequency array, for Nash limit: (-1)^(c+1)mgl/Iw
    params : dict
        parameters dictionary
    dispersion : array or list
        dispersion relation of...
    rm_images : bool
        Whether or not to delete all the images after a movie has been made of the DOS
    gapims_only : bool
        Whether to just plot modes near the middle of the DOS frequency range
    save_into_subdir: bool
        Whether to save the movie in the same sudir where the DOS/ directory (containing the frames) is placed.
    """
    glat = gyro_lattice
    NP = len(glat.lattice.xy)
    print 'Getting eigenvals/vects of dynamical matrix...'
    # Find eigval/vect
    eigval, eigvect = glat.get_eigval_eigvect()
    OmK = glat.OmK
    Omg = glat.Omg

    # prepare DOS output dir
    if datadir == 'auto':
        datadir = glat.lp['meshfn']

    if 'meshfn_exten' in glat.lp:
        exten = glat.lp['meshfn_exten']
    elif glat.lp['dcdisorder']:
        # there should be meshfn_exten defined in lp, but since there is not, we make it here
        if glat.lp['V0_pin_gauss'] > 0:
            exten = '_pinV' + sf.float2pstr(glat.lp['V0_pin_gauss']) + \
                    '_strV' + sf.float2pstr(glat.lp['V0_spring_gauss'])
        elif glat.lp['V0_pin_flat'] > 0:
            exten = '_pinVf' + sf.float2pstr(glat.lp['V0_pin_flat']) + \
                    '_strVf' + sf.float2pstr(glat.lp['V0_spring_flat'])
    else:
        exten = ''

    DOSdir = datadir + 'DOS' + exten + '/'
    dio.ensure_dir(DOSdir)

    #####################################
    # Prepare for plotting
    #####################################
    print 'Preparing plot settings...'
    omg, omk, do_bc_determined, bcstr, btmp, ctmp, pin = determine_bc_pin(Omg, OmK)
    do_bc = do_bc and do_bc_determined

    # Options for how to color the DOS header
    if color == 'pr':
        ipr = glat.get_ipr()
        colorV = 1./ipr
        lw = 0
        cbar_label = r'$p$'
        colormap = 'viridis_r'
    elif color == 'ipr':
        ipr = glat.get_ipr()
        colorV = ipr
        lw = 0
        cbar_label = r'$p^{-1}$'
        colormap = 'viridis'
    elif color == 'ill':
        ill = glat.get_ill()
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
    fig, DOS_ax, eig_ax = leplt.initialize_eigvect_DOS_header_plot(eigval, glat.lattice.xy, sim_type=sim_type,
                                                                   colorV=colorV, colormap=colormap, linewidth=lw,
                                                                   cax_label=cbar_label, cbar_nticks=2,
                                                                   cbar_tickfmt='%0.3f')

    dostitle = r'$\Omega_g = $' + '{0:0.2f}'.format(glat.lp['Omg'])
    if glat.lp['ABDelta']:
        dostitle += r'$\pm$' + '{0:0.2f}'.format(glat.lp['ABDelta'])
    dostitle += r' $\Omega_k=$' + '{0:0.2f}'.format(glat.lp['Omk'])
    DOS_ax.set_title(dostitle)
    glat.lattice.plot_BW_lat(fig=fig, ax=eig_ax, save=False, close=False, axis_off=False, title='')

    # Make strings for spring, pin, k, and g values
    if do_bc:
        springstr_Hz = '{0:.03f}'.format(omk[0]/(2.*np.pi))
        pinstr_Hz = '{0:.03f}'.format(omg[0]/(2.*np.pi))
    else:
        springstr_Hz = ''
        pinstr_Hz = ''

    if springstr_Hz != '' and pinstr_Hz != '':
        text2show = 'spring = ' + springstr_Hz + ' Hz,  pin = ' + pinstr_Hz + ' Hz\n' + '\n' + bcstr
        fig.text(0.4, 0.1, text2show, horizontalalignment='center', verticalalignment='center')

    # Add schematic of hanging/standing top spinning with dir
    if do_bc:
        schem_ax = plt.axes([0.85, 0.0, .025*5, .025*7], axisbg='w')
        # drawing
        schem_ax.plot([0., 0.2], [1-btmp, btmp], 'k-' )
        schem_ax.scatter([0.2], [btmp], s=150, c='k')
        schem_ax.arrow(0.2, btmp, -(-1)**ctmp*0.06, 0.3*(-1)**(btmp+ctmp),
                       head_width=0.3, head_length=0.1, fc='b', ec='b')
        wave_x = np.arange(-0.07*5, 0.0, 0.001)
        wave_y = 0.1*np.sin(wave_x*100)+1.-btmp
        schem_ax.plot(wave_x, wave_y, 'k-')
        schem_ax.set_xlim(-0.1*5, .21*5)
        schem_ax.set_ylim(-0.1*7, .21*7)
        schem_ax.axis('off')

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
                    # glat.lattice.plot_BW_lat(fig=fig, ax=DOS_ax, save=False, close=False, axis_off=False, title='')
                    fig, [scat_fg, scat_fg2, p, f_mark, lines_12_st], cw_ccw = \
                        leplt.construct_eigvect_DOS_plot(glat.lattice.xy, fig, DOS_ax, eig_ax, eigval, eigvect,
                                                         ii, sim_type, glat.lattice.NL, glat.lattice.KL,
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


def determine_bc_pin(Omg, OmK):
    # Determine b,c
    omk = OmK[np.where(abs(OmK) > 0.)[0], np.where(abs(OmK) > 0)[1]]
    omg = Omg.ravel()[np.where(abs(Omg.ravel()) > 0.)[0]]
    if len(omg) == 0:
        omg = np.zeros(len(Omg), dtype=float)

    # omk > 0, omg > 0: b=0,c=1
    # omk > 0, omg < 0: b=1,c=0
    # omk < 0, omg > 0: b=1,c=1
    # omk < 0, omg < 0: b=0,c=0
    # Check if uniform/homogenous omk and omg
    # and make string for labelling spin direction and whether
    # pendulum is standing or hanging.
    if (omk == omk[0]).all():
        if (omg == omg[0]).all():
            do_bc = True
            # Find values for b and c
            if omk[0] > 0:
                if omg[0] > 0:
                    bstr = 'b=0'
                    cstr = 'c=1'
                    btmp = 0
                    ctmp = 1
                elif omg[0] < 0:
                    bstr = 'b=1'
                    cstr = 'c=0'
                    btmp = 1
                    ctmp = 0
                else:
                    bstr = 'b=gravity is off'
                    cstr = 'c=gravity is off'
                    btmp = 0
                    ctmp = 1
            elif omk[0] < 0:
                if omg[0] > 0:
                    bstr = 'b=1'
                    cstr = 'c=1'
                    btmp = 1
                    ctmp = 1
                elif omg[0] < 0:
                    bstr = 'b=0'
                    cstr = 'c=0'
                    btmp = 0
                    ctmp = 0
                else:
                    bstr = 'b=gravity is off'
                    cstr = 'c=gravity is off'
                    btmp = 0
                    ctmp = 0
        else:
            do_bc = False
            bstr = ''
            cstr = ''
    else:
        do_bc = False
        bstr = ''
        cstr = ''

    bcstr = bstr + ', '+ cstr

    if do_bc:
        # temporarily store the homogeneous pin frequency as pin
        pin = omg[0]
    else:
        pin = -5000
        btmp = ''
        ctmp = ''

    return omg, omk, do_bc, bcstr, btmp, ctmp, pin


def make_movie(imgname, movname, indexsz='05', framerate=10, imgdir=None, rm_images=False, save_into_subdir=False):
    """Create a movie from a sequence of images. Options allow for deleting folder automatically after making movie.
    Will run './ffmpeg', '-framerate', str(int(framerate)), '-i', imgname + '%' + indexsz + 'd.png', movname + '.mov',
         '-vcodec', 'libx264', '-profile:v', 'main', '-crf', '12', '-threads', '0', '-r', '100', '-pix_fmt', 'yuv420p'])

    Parameters
    ----------
    imgname : str
        path and filename for the images to turn into a movie
    movname : str
        path and filename for output movie
    indexsz : str
        string specifier for the number of indices at the end of each image (ie 'file_000.png' would merit '03')
    framerate : int (float may be allowed)
        The frame rate at which to write the movie
    imgdir : str or None
        name of subdirectory to delete if rm_images and save_into_subdir are both True, ie folder containing the images
        Note: this is not the full path if save_into_subir is False.
    rm_images : bool
        Remove the images from disk after writing to movie
    save_into_subdir : bool
        The images are saved into a folder which can be deleted after writing to a movie, if rm_images is True and
        imgdir is not None (ie images are not on same heirarchical level as movie or other data)
    """
    # Convert indexsz to a string if not already one
    if isinstance(indexsz, int):
        indexsz = str(indexsz)
    elif isinstance(indexsz, float):
        indexsz = str(int(indexsz))

    if movname[-4:] != '.mov':
        movname += '.mov'

    call_list = ['./ffmpeg', '-framerate', str(int(framerate)), '-i', imgname + '%' + indexsz + 'd.png',
                 movname, '-vcodec', 'libx264', '-profile:v', 'main', '-crf', '12', '-threads', '0',
                 '-r', '100', '-pix_fmt', 'yuv420p']
    print 'calling: ', call_list
    subprocess.call(call_list)

    # Delete the original images
    if rm_images:
        print 'Deleting the original images...'
        if save_into_subdir and imgdir is not None:
            print 'Deleting folder ' + imgdir
            subprocess.call(['rm', '-r', imgdir])
        else:
            print 'Deleting folder contents ' + imgname + '*.png'
            subprocess.call(['rm', imgname + '*.png'])


def save_normal_modes_haldane(haldane_lattice, datadir='auto', rm_images=True, gapims_only=False,
                              save_into_subdir=False, overwrite=True, color='pr'):
    """
    Plot the normal modes of a haldane model lattice with fixed pivot points and make a movie of them.

    Parameters
    ----------
    haldane_lattice : HaldaneLattice() instance
        the complex-NNN hopping lattice for which to plot the normal modes
    datadir: string
        directory where simulation data is stored
    rm_images : bool
        Whether or not to delete all the images after a movie has been made of the DOS
    gapims_only : bool
        Whether to just plot modes near the middle of the DOS frequency range
    save_into_subdir: bool
        Whether to save the movie in the same sudir where the DOS/ directory (containing the frames) is placed.
    """
    hlat = haldane_lattice
    NP = len(hlat.lattice.xy)
    print 'Getting eigenvals/vects of dynamical matrix...'
    # Find eigval/vect
    eigval, eigvect = hlat.get_eigval_eigvect()
    t2 = hlat.t2
    pin = hlat.pin

    # prepare DOS output dir
    if datadir == 'auto':
        datadir = hlat.lp['meshfn']

    if 'meshfn_exten' in hlat.lp:
        exten = hlat.lp['meshfn_exten']
    elif hlat.lp['dcdisorder']:
        raise RuntimeError('lemov: Since dcdisorder is True, there should be a meshfn_exten in lp')
        # there should be meshfn_exten defined in lp, but since there is not, we make it here
        # exten = 'pin_mean' + sf.float2pstr(hlat.lp['pin']) + \
        #         '_pinV' + sf.float2pstr(hlat.lp['V0_pin_gauss']) + \
        #         '_strV' + sf.float2pstr(hlat.lp['V0_spring_gauss'])
    else:
        exten = ''

    DOSdir = datadir + 'DOS_haldane' + exten + '/'
    dio.ensure_dir(DOSdir)

    #####################################
    # Prepare for plotting
    #####################################
    print 'Preparing plot settings...'
    # Options for how to color the DOS header
    if color == 'pr':
        ipr = hlat.get_ipr()
        colorV = 1./ipr
        lw = 0
        cbar_label = r'$p$'
        colormap = 'viridis_r'
    elif color == 'ipr':
        ipr = hlat.get_ipr()
        colorV = ipr
        lw = 0
        cbar_label = r'$p^{-1}$'
        colormap = 'viridis'
    elif color == 'ill':
        ill = hlat.get_ill()
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
    fig, DOS_ax, eig_ax = leplt.initialize_eigvect_DOS_header_plot(eigval, hlat.lattice.xy, sim_type='haldane',
                                                                   colorV=colorV, colormap=colormap,
                                                                   linewidth=lw, cax_label=cbar_label, cbar_nticks=2,
                                                                   xlabel_pad=10, ylabel_pad=10,
                                                                   cbar_tickfmt='%0.3f')

    dostitle = r'$V = $' + '{0:0.2f}'.format(hlat.lp['pin'])
    if hlat.lp['ABDelta']:
        dostitle += r'$\pm$' + '{0:0.2f}'.format(hlat.lp['ABDelta'])
    elif hlat.lp['V0_pin_gauss']:
        dostitle += r'$\pm \sigma$, with $\sigma = $' + '{0:0.2f}'.format(hlat.lp['V0_pin_gauss'])

    dostitle += r' $t_1=$' + '{0:0.2f}'.format(hlat.lp['t1'])
    if abs(hlat.lp['t2a']) > 1e-7:
        dostitle += r' $t_2=$' + '{0:0.2f}'.format(hlat.lp['t2a']) + r'+$i$' + '{0:0.2f}'.format(hlat.lp['t2'])
    else:
        dostitle += r' $t_2=$' + '{0:0.2f}'.format(hlat.lp['t2'])
    DOS_ax.set_title(dostitle)
    hlat.lattice.plot_BW_lat(fig=fig, ax=eig_ax, save=False, close=False, axis_off=False, title='')

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
                movname += names[ii] + '/' + names[ii] + '_DOS' + haldane_lattice.lp['meshfn_exten']
            else:
                movname += names[ii]

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
                todo = range(int(len(eigval)))

            if done_pngs < len(todo):
                dmyi = 0
                for ii in todo:
                    if np.mod(ii, 50) == 0:
                        print 'plotting eigvect ', ii, ' of ', len(eigval)
                    # hlat.lattice.plot_BW_lat(fig=fig, ax=DOS_ax, save=False, close=False, axis_off=False, title='')
                    fig, [scat_fg, scat_fg2, p, f_mark, lines_12_st], cw_ccw = \
                        hlatpfns.construct_haldane_eigvect_DOS_plot(hlat.lattice.xy, fig, DOS_ax, eig_ax, eigval,
                                                                    eigvect, ii, hlat.lattice.NL, hlat.lattice.KL,
                                                                    marker_num=0, color_scheme='default')
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


def save_normal_modes_mass(mass_lattice, datadir='auto', rm_images=True,
                           save_into_subdir=False, overwrite=True, color='pr', start_number=0):
    """
    Plot the normal modes of a coupled gyros with fixed pivot points and make a movie of them.
    Note that b and c are built INTO the signs of OmK and Omg.
    --> b (0 -> 'hang',  1 -> 'stand').
    --> c (0 -> aligned with a,  1 -> aligned with a)

    Parameters
    ----------
    mass_lattice : MassLattice instance
        with attributes including .lattice, .lp, etc
    datadir: string
        directory where simulation data is stored
    rm_images : bool
        Whether or not to delete all the images after a movie has been made of the DOS
    gapims_only : bool
        Whether to just plot modes near the middle of the DOS frequency range
    save_into_subdir: bool
        Whether to save the movie in the same sudir where the DOS/ directory (containing the frames) is placed.
    """
    mlat = mass_lattice
    NP = len(mlat.lattice.xy)
    print 'Getting eigenvals/vects of dynamical matrix...'
    # Find eigval/vect
    eigval, eigvect = mlat.get_eigval_eigvect()
    kk = mlat.kk
    mass = mlat.mass

    # prepare DOS output dir
    if datadir == 'auto':
        datadir = mlat.lp['meshfn']

    if 'meshfn_exten' in mlat.lp:
        exten = mlat.lp['meshfn_exten']
    elif mlat.lp['dcdisorder']:
        # there should be meshfn_exten defined in lp, but since there is not, we make it here
        if 'V0_pin_gauss' in mlat.lp:
            if mlat.lp['V0_pin_gauss'] > 0:
                exten = '_pinV' + sf.float2pstr(mlat.lp['V0_pin_gauss']) + \
                        '_strV' + sf.float2pstr(mlat.lp['V0_spring_gauss'])
        if 'V0_pin_flat' in mlat.lp:
            if mlat.lp['V0_pin_flat'] > 0:
                exten = '_pinVf' + sf.float2pstr(mlat.lp['V0_pin_flat']) + \
                        '_strVf' + sf.float2pstr(mlat.lp['V0_spring_flat'])
    else:
        exten = ''

    DOSdir = datadir + 'DOS' + exten + '/'
    dio.ensure_dir(DOSdir)

    #####################################
    # Prepare for plotting
    #####################################
    # Options for how to color the DOS header
    if color == 'pr':
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
    # print 'lemov: eigval = ', eigval
    # plt.plot(np.arange(len(eigval)), eigval, '.-')
    # plt.show()
    # sys.exit()
    fig, DOS_ax, eig_ax = leplt.initialize_eigvect_DOS_header_plot(eigval, mlat.lattice.xy, sim_type='mass',
                                                                   colorV=colorV, colormap=colormap, linewidth=lw,
                                                                   cax_label=cbar_label, cbar_nticks=2,
                                                                   cbar_tickfmt='%0.3f')

    dostitle = r'$m = $' + '{0:0.2f}'.format(mlat.lp['mass'])
    if mlat.lp['ABDelta']:
        dostitle += r'$\pm$' + '{0:0.2f}'.format(mlat.lp['ABDelta'])
    dostitle += r' $k=$' + '{0:0.2f}'.format(mlat.lp['kk'])
    DOS_ax.set_title(dostitle)
    mlat.lattice.plot_BW_lat(fig=fig, ax=eig_ax, save=False, close=False, axis_off=False, title='')

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
            todo = range(int(round(len(eigval))))

            if done_pngs < len(todo):
                dmyi = 0
                for ii in todo:
                    if np.mod(ii, 50) == 0:
                        print 'plotting eigvect ', ii, ' of ', len(eigval)
                    # mlat.lattice.plot_BW_lat(fig=fig, ax=DOS_ax, save=False, close=False, axis_off=False, title='')
                    fig, [scat_fg, scat_fg2, p, f_mark, lines_12_st], cw_ccw = \
                        leplt.construct_eigvect_DOS_plot(mlat.lattice.xy, fig, DOS_ax, eig_ax, eigval, eigvect,
                                                         ii, 'mass', mlat.lattice.NL, mlat.lattice.KL,
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
                     '-crf', '12', '-threads', '0', '-r', '100', '-pix_fmt', 'yuv420p', '-start_number',
                     str(start_number)])

    if rm_images:
        # Delete the original images
        if not save_into_subdir:
            print 'Deleting folder ' + DOSdir
            subprocess.call(['rm', '-r', DOSdir])
        else:
            print 'Deleting folder contents ' + DOSdir + 'DOS_*.png'
            subprocess.call(['rm', '-r', DOSdir + 'DOS_*.png'])
