from numpy import *
import matplotlib.pyplot as plt
import cPickle as pickle
import time
import os.path
import chern_functions_gen as cf
import lattice_functions as lf
import view_plot as vp
import sys

if __name__ == '__main__':
    print 'running example.'
    base_dir = '/Users/npmitchell/Desktop/test'
    tvals = [0.86]  # initialize the Omega_k values for this lattice.
    # Some lattices might have springs with different constants so this list could be longer.

    diff = 0.0
    omg = 0.98
    ons = [omg + diff, omg - diff]  # Omega_g values for the A&B sublattices in this system.

    # I will be showing an example of a honeycomb lattice
    delta = 120
    phi = 0
    min_density = 400  # number of points per area in BZ for calculation.

    # this is really the only function you need to change to do a different kind of lattice.
    mM, vertex_points, od, fn, ar = lf.honeycomb_sheared(tvals, delta, phi, ons, base_dir)

    # set the directory for saving images and data.  The function above actually creates these directories for you.
    #  I know this isn't exaclty ideal.
    od_im = od + '/images/'
    od_dat = od + '/data/'

    print 'created lattice'
    bzarea = cf.PolygonArea(vertex_points)

    # I usually check to see if the data is already there.  New data is appended to the old data.
    if os.path.isfile(od_dat + '/data_dict_' + fn + '.pickle'):

        of = open(od_dat + '/data_dict_' + fn + '.pickle', 'rb')
        data = pickle.load(of)

        bands = list(data['bands'])
        kkx = list(data['kx'])
        kky = list(data['ky'])
        traces = list(data['traces'])

        of.close()
    else:
        kkx = []
        kky = []
        bands = []
        traces = []

    NP = int(min_density * bzarea)

    max_x = max(abs(vertex_points[:, 0]))
    max_y = max(abs(vertex_points[:, 1]))

    kx, ky = cf.generate_kx_ky_random(NP, vertex_points, crx=max_x + 0.25, cry=max_y + 0.25)

    if not len(kkx) > 2 * NP:
        start_time = time.time()
        for i in range(len(kx)):

            tpi = []

            if i % 500 == 0:
                end_time = time.time()
                tpi.append(abs((start_time - end_time) / (i + 1)))
                total_time = mean(tpi) * len(kx)
                print 'Estimated time remaining', '%0.2f s' % (total_time - (end_time - start_time))

            Kx = kx[i]
            Ky = ky[i]

            kkx.append(Kx)
            kky.append(Ky)

            eigval, tr = cf.calc_bands(mM, Kx, Ky)
            traces.append(tr)
            bands.append(eigval)

    bands = array(bands)
    traces = array(traces)
    kkx = array(kkx)
    kky = array(kky)

    bv = []
    print shape(bands)
    print shape(traces)
    for i in range(len(bands[0])):
        b = ((1j / (2 * pi)) * mean(traces[:, i]) * bzarea)  # chern numbers for bands
        bv.append(b)

    ax_lat, cc, lc, fig = vp.save_plot(kkx, kky, bands, traces, tvals[:ar[0]], ons[:ar[1]], vertex_points,
                                       od=od_im + fn)
    R, Ni, Nk, cols, line_cols = lf.honeycomb_sheared_vis(delta, phi)
    lf.lattice_plot(R, Ni, Nk, ax_lat, cols, line_cols)
    plt.savefig(od_im + fn + '.png')
    # plt.show()
    plt.close()

    dat_dict = {'tvals': tvals, 'ons': ons, 'kx': kkx, 'ky': kky, 'chern': bv, 'bands': bands, 'traces': traces,
                'vertex_points': vertex_points}
    cf.save_pickled_data(od_dat, '/data_dict_' + fn, dat_dict)
    print 'saved', fn, 'with vertex_points'
