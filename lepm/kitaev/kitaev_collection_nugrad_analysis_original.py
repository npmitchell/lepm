# from scipy import ndimage
# check = False
# # Measure localization to where gradients in chern values are high after loading varyloc cherns
# cmaps.register_colormaps()
# # For a single lattice, collate cherns with varying omegac
# meshfn, xyffind = le.build_meshfn(lp)
# lp['meshfn'] = meshfn
# lat = lattice_class.Lattice(lp)
# # try:
# lat.load()
# # except IOError:
# #     lat.build()
# #     lat.save()
# glat = gyro_lattice_class.GyroLattice(lat, lp)
# glat.load()
# gc = gyro_collection.GyroCollection()
# gc.add_gyro_lattice(glat)
# print 'Creating chern collection from gyro_collection...'
# if args.Nks == 211:
# cp['ksize_frac_arr'] = np.arange(0.0, 1.0500001, 0.005)
# fps = 20
# elif args.Nks == 201:
# cp['ksize_frac_arr'] = np.arange(0.0, 0.5000001, 0.0025)
# fps = 20
# elif args.Nks == 121:
# cp['ksize_frac_arr'] = np.arange(0.0, 0.3000001, 0.0025)
# fps = 12
# elif args.Nks == 30:
# cp['ksize_frac_arr'] = np.arange(0.0, 0.30, 0.01)
# fps = 3
# elif args.Nks == 31:
# cp['ksize_frac_arr'] = np.arange(0.0, 0.31, 0.01)
# fps = 3
#
# kcoll = KitaevCollection(gc, cp=cp)
# print "kcoll.cp['ksize_frac_arr'] = ", kcoll.cp['ksize_frac_arr']
# print 'Getting chern calculations with varying position, holding omegac constant:\n omegac=', cp['omegac'][0]
#
# # Load or compute all cherns
# kcoll.calc_cherns_varyloc(step=args.step, verbose=False)
#
# # For each glat, for every ksize, compute gradients. Then look at evects.
# for glat_name in kcoll.cherns:
# glat = kcoll.cherns[glat_name][0].gyro_lattice
#
# outdir = kitaev_functions.get_ccpath(cp, lp, rootdir='/Users/npmitchell/Dropbox/Soft_Matter/GPU/',
#                                      method='varyloc')
# le.ensure_dir(le.prepdir(outdir))
# outdir = le.prepdir(outdir) + glat.lp['LatticeTop']
# outfn = outdir + '_gnu_dict.pkl'
# if glob.glob(outfn):
#     print 'Loading collected gnu results instead of computing/loading cherns...'
#     with open(outfn, 'rb') as fn:
#         gnudict = pickle.load(fn)
#
#     local = gnudict['gnu_psi2']
#     xyvec = gnudict['xyvec']
#     ksizes = gnudict['ksizes']
#     kszf = gnudict['kszf']
#     nugrids = gnudict['nugrids']
#     nulap = gnudict['nulap']
#     nugsum = gnudict['nugsum']
#     local2 = gnudict['local2']
#     local3 = gnudict['local3']
#     local4 = gnudict['local4']
#     local5 = gnudict['local5']
#     local5 = gnudict['local6']
#     eigval = gnudict['eigval']
#
#     # Make nugrad from nugrids
#     ind = 0
#     nugrad = np.zeros_like(nugrids, dtype=float)
#     nugrids_cut = np.zeros_like(nugrids, dtype=float)
#     nugrad_cut = np.zeros_like(nugrids, dtype=float)
#     for nugrid in nugrids:
#         # For each ksize, get 2d gradient magnitude and find ww region (where grad is large)
#         grady, gradx = np.gradient(nugrid)
#         nugrad[ind, :, :] = np.sqrt(grady**2 + gradx**2)
#
#         # Introduce cutoff
#         nugrids_cut[ind] = nugrid
#         nugrids_cut[ind][nugrid > 0.2] = 1.0
#         nugrids_cut[ind][nugrid < -0.2] = -1.0
#         grady, gradx = np.gradient(nugrid)
#         nugrad_cut[ind, :, :] = np.sqrt(grady**2 + gradx**2)
#         ind += 1
#
# else:
#
#     # Assume all cherns have the same len(ksize), so preallocate array
#     # Make xynu, for which xynu[:,i] is the map of chern vals for the ith ksize
#     # also make xyvec, so that xyvec[i] = [x, y] for computation xynu[i,:]
#     kszf = kcoll.cherns[glat_name][0].chern_finsize[:, 1]
#     xynu = np.zeros((len(kcoll.cherns[glat_name]), len(kszf)), dtype=float)
#     ksizes = np.zeros((len(kcoll.cherns[glat_name]), len(kszf)), dtype=float)
#     xyvec = np.zeros((np.shape(xynu)[0], 2), dtype=float)
#     ind = 0
#     for chernii in kcoll.cherns[glat_name]:
#         # Add this chern to array of stored vals
#         xx = float(chernii.cp['poly_offset'].split('/')[0])
#         yy = float(chernii.cp['poly_offset'].split('/')[1])
#         xynu[ind] = np.real(chernii.chern_finsize[:, -1])
#         ksizes[ind] = chernii.chern_finsize[:, 3]
#         xyvec[ind, :] = np.array([xx, yy])
#         ind += 1
#
#     if check:
#         print 'xynu = ', xynu
#         plt.imshow(xynu)
#         plt.show()
#
#     # Make grid of nu values, in fact one for each ksize
#     # nugrids[i] is the grid of nu values for the ith ksize
#     lenx = int(np.sqrt(len(kcoll.cherns[glat_name])))
#     print 'lenx = ', lenx
#     print 'len(kcoll.cherns[glat_name]) = ', len(kcoll.cherns[glat_name])
#     nugrids = xynu.reshape((lenx, lenx, len(kszf))).T
#
#     if check:
#         print 'np.shape(xynu) = ', np.shape(xynu)
#         tmp = nugrids[15, :, :]
#         mappable = plt.imshow(tmp, interpolation='nearest', cmap='rwb0', vmin=-1, vmax=1)
#         plt.colorbar()
#         plt.title('chern values in space')
#         plt.show()
#
#     nugrad = np.zeros_like(nugrids, dtype=float)
#     nulap = np.zeros_like(nugrids, dtype=float)
#     nugrids_cut = np.zeros_like(nugrids, dtype=float)
#     nugrad_cut = np.zeros_like(nugrids, dtype=float)
#     nugsum = np.zeros_like(nugrids, dtype=float)
#     ind = 0
#     for nugrid in nugrids:
#         # For each ksize, get 2d gradient magnitude and find ww region (where grad is large)
#         grady, gradx = np.gradient(nugrid)
#         nugrad[ind, :, :] = np.sqrt(grady**2 + gradx**2)
#         laplace = 1./6. * np.array([[1, 4, 1], [4, -20, 4], [1, 4, 1]])
#         nulap[ind, :, :] = ndimage.convolve(nugrid, laplace, mode='reflect')
#
#         diff = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]])
#         nugsum[ind, :, :] = 0.25 * np.abs(ndimage.convolve(nugrid, diff, mode='reflect'))
#         diff = np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]])
#         nugsum[ind, :, :] += 0.25 * np.abs(ndimage.convolve(nugrid, diff, mode='reflect'))
#         diff = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]])
#         nugsum[ind, :, :] += 0.25 * np.abs(ndimage.convolve(nugrid, diff, mode='reflect'))
#         diff = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]])
#         nugsum[ind, :, :] += 0.25 * np.abs(ndimage.convolve(nugrid, diff, mode='reflect'))
#
#         # Introduce cutoff
#         nugrids_cut[ind] = nugrid
#         nugrids_cut[ind][nugrid > 0.2] = 1.0
#         nugrids_cut[ind][nugrid < -0.2] = -1.0
#         grady, gradx = np.gradient(nugrids_cut[ind])
#         nugrad_cut[ind, :, :] = np.sqrt(grady**2 + gradx**2)
#
#         if check:
#             print 'np.shape(nugrid) = ', np.shape(nugrid)
#             leplt.plot_pcolormesh(xyvec[:, 0], xyvec[:, 1], nugrid.T.ravel(), 100,
#                                   method='nearest', cmap='rwb0',
#                                   vmin=-1.0, vmax=1.0, xlabel='x', ylabel='y',  cax_label=r'$\nu$',
#                                   fontsize=12)
#             le.movie_plot_2D(glat.lattice.xy, glat.lattice.BL, 0 * glat.lattice.BL[:, 0],
#                              None, None, ax=plt.gca(), axcb=None, bondcolor='k',
#                              colorz=False, ptcolor=None, figsize='auto',
#                              colormap='BlueBlackRed', bgcolor='#ffffff', axis_off=False, axis_equal=True,
#                              lw=0.2)
#             kpfns.add_kitaev_regions_to_plot(kcoll.cherns[glat_name][0], ind, offsetxy=np.array([0, 0]))
#             plt.pause(0.1)
#             plt.clf()
#             leplt.plot_pcolormesh(xyvec[:, 0], xyvec[:, 1], nugrad[ind].T.ravel(), 100,
#                                   method='nearest', cmap='viridis',
#                                   vmin=0.0, vmax=None, xlabel='x', ylabel='y',  cax_label=r'$|\nabla\nu|$',
#                                   fontsize=12)
#             le.movie_plot_2D(glat.lattice.xy, glat.lattice.BL, 0 * glat.lattice.BL[:, 0],
#                              None, None, ax=plt.gca(), axcb=None, bondcolor='k',
#                              colorz=False, ptcolor=None, figsize='auto',
#                              colormap='BlueBlackRed', bgcolor='#ffffff', axis_off=False, axis_equal=True,
#                              lw=0.2)
#             kpfns.add_kitaev_regions_to_plot(kcoll.cherns[glat_name][0], ind, offsetxy=np.array([0, 0]))
#             plt.pause(0.1)
#             print 'np.shape(nugrad[ind]) = ', np.shape(nugrad[ind])
#             print 'nugrad[ind,:,:] = ', nugrad[ind, :, :]
#             # plt.imshow(nugrid, interpolation='none', cmap='rwb0', vmin=-1, vmax=1)
#             # plt.pause(0.1)
#             # plt.clf()
#             # plt.imshow(nugrad[ind, :, :], interpolation='none', cmap='rwb0', vmin=-1, vmax=1)
#             # plt.pause(0.4)
#             plt.clf()
#
#         ind += 1
#
#     # convert back to same order as xyvec for later
#     # gnu_vecs[:,i] is the vector of chern gradient magnitudes for the ith ksize, assoc with xyvec
#     gnu_vecs = nugrad.T.reshape(-1, len(kszf))
#     gcnu_vecs = nugrad_cut.T.reshape(-1, len(kszf))
#     lnu_vecs = nulap.T.reshape(-1, len(kszf))
#     gsnu_vecs = nugsum.T.reshape(-1, len(kszf))
#
#     # Smooth gradient
#     snugsum = np.zeros_like(nugsum, dtype=float)
#     for ind in range(len(nugsum)):
#         snugsum[ind] = ndimage.uniform_filter(nugsum[ind], (2, 2))
#     sgsnu_vecs = snugsum.T.reshape(-1, len(kszf))
#
#     if check:
#         print 'gnu_vecs = ', gnu_vecs
#         leplt.plot_pcolormesh(xyvec[:, 0], xyvec[:, 1], gnu_vecs[:, 15], 100, method='nearest', cmap='viridis',
#                               vmin=0., vmax=1.0, xlabel='x', ylabel='y',  cax_label=r'$|\nabla\nu$|',
#                               fontsize=12)
#         plt.show()
#
#     if check:
#         plt.imshow(nugrids[15, :, :], interpolation='none', cmap='rwb0', vmin=-1, vmax=1)
#         plt.title('nu')
#         plt.show()
#
#         plt.imshow(nugrad[15, :, :], interpolation='none', cmap='viridis', vmin=0, vmax=1)
#         plt.title('gradient')
#         plt.show()
#
#     # Associate each particle with an xy region
#     # pt2loc's ith element is the index of xynu or nugrad assoc with ith gyro
#     pt2loc = np.zeros_like(glat.lattice.xy[:, 0], dtype=int)
#     ind = 0
#     for xy in glat.lattice.xy:
#         dist = np.abs(xyvec - xy)[:, 0]**2 + np.abs(xyvec - xy)[:, 1]**2
#         loc = np.argmin(dist)
#         pt2loc[ind] = loc
#         ind += 1
#
#     # Look at evects for this network. Compute L for each evect. Plot L as a function of evals.
#     eigval, eigvect = glat.load_eigval_eigvect(attribute=True)
#     ind = 0
#     local = np.zeros((len(kszf), len(eigval)/2), dtype=float)
#     local2 = np.zeros((len(kszf), len(eigval)/2), dtype=float)
#     local3 = np.zeros((len(kszf), len(eigval)/2), dtype=float)
#     local4 = np.zeros((len(kszf), len(eigval)/2), dtype=float)
#     local5 = np.zeros((len(kszf), len(eigval)/2), dtype=float)
#     local6 = np.zeros((len(kszf), len(eigval)/2), dtype=float)
#     for ind in range(int(len(eigval)*0.5)):
#         if ind % 100 == 0:
#             print 'Calculating locality for eigval ', ind, ' of ', len(eigval), '...'
#         # Eigvect is stored as NModes x NP*2 array, with x and y components alternating, like:
#         # x0, y0, x1, y1, ... xNP, yNP.
#         mag1 = eigvect[ind]
#         mag1x = np.array([mag1[2*i] for i in range(len(mag1)/2)])
#         mag1y = np.array([mag1[2*i+1] for i in range(len(mag1)/2)])
#
#         # Get magnitude of displacements
#         mag2 = np.array([abs(mag1x[i])**2 + abs(mag1y[i])**2 for i in range(len(mag1x))]).flatten()
#         local[:, ind] = np.array([np.sum(mag2 * gnu_vecs[pt2loc, kk]) / np.sum(mag2)
#                                   for kk in range(len(kszf))])
#         local2[:, ind] = np.array([np.sum(mag2 * lnu_vecs[pt2loc, kk]) / np.sum(mag2)
#                                   for kk in range(len(kszf))])
#         local3[:, ind] = np.array([np.sum(mag2 * gcnu_vecs[pt2loc, kk]) / np.sum(mag2)
#                                   for kk in range(len(kszf))])
#         local4[:, ind] = np.array([np.sum(mag2 * np.abs(lnu_vecs[pt2loc, kk])) / np.sum(mag2)
#                                   for kk in range(len(kszf))])
#         # local5 is gradsum of nu
#         local5[:, ind] = np.array([np.sum(mag2 * np.abs(gsnu_vecs[pt2loc, kk])) / np.sum(mag2)
#                                   for kk in range(len(kszf))])
#         local6[:, ind] = np.array([np.sum(mag2 * np.abs(sgsnu_vecs[pt2loc, kk])) / np.sum(mag2)
#                                   for kk in range(len(kszf))])
#     # Save results
#     lld = {'gnu_psi2': local, 'ksizes': ksizes, 'kszf': kszf, 'xyvec': xyvec, 'nugrids': nugrids,
#            'local2': local2, 'local3': local3, 'local4': local4, 'local5': local5, 'local6': local6,
#            'eigval': eigval, 'nulap': nulap, 'nugsum': nugsum }
#     outfn = outdir + '_gnu_dict.pkl'
#     with open(outfn, 'wb') as fn:
#         pickle.dump(lld, fn)
#
# # Now plot results
# print 'Plotting results...'
# import matplotlib.colors
# copper = plt.get_cmap('copper_r')
# cNorm = matplotlib.colors.Normalize(vmin=0, vmax=len(local))
# copperMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=copper)
# jet = plt.get_cmap('spectral')
# cNorm_ev = matplotlib.colors.Normalize(vmin=1.0, vmax=max(np.imag(eigval)))
# jetMap = matplotlib.cm.ScalarMappable(norm=cNorm_ev, cmap=jet)
#
# top = range(len(local[0]))
# print 'set outdir = ', outdir
# le.ensure_dir(outdir)
#
# # pick a representative index for a later stage of ksize
# repi = int(len(ksizes[0]) * 0.9)
# repi2 = int(len(ksizes[0]) * 0.4)
# sz = 1
#
# plt.clf()
# ind = 0
# leplt.plot_pcolormesh(xyvec[:, 0], xyvec[:, 1], nugrids[repi].T.ravel(), 100,
#                       method='nearest', cmap='rwb0',
#                       vmin=-1.0, vmax=1.0, xlabel='x', ylabel='y',  cax_label=r'$\nu$',
#                       fontsize=12)
# le.movie_plot_2D(glat.lattice.xy, glat.lattice.BL, 0 * glat.lattice.BL[:, 0],
#                  None, None, ax=plt.gca(), axcb=None, bondcolor='k',
#                  colorz=False, ptcolor=None, figsize='auto',
#                  colormap='BlueBlackRed', bgcolor='#ffffff', axis_off=False, axis_equal=True,
#                  lw=0.2)
# kpfns.add_kitaev_regions_to_plot(kcoll.cherns[glat_name][0], repi, offsetxy=np.array([15, 0]))
# plt.title(r'$\nu$')
# plt.ylabel(r'$y$')
# plt.xlabel(r'$x$')
# plt.savefig(outdir + '_nugrid.png')
#
# plt.clf()
# leplt.plot_pcolormesh(xyvec[:, 0], xyvec[:, 1], nugrad[repi].T.ravel(), 100,
#                       method='nearest', cmap='viridis',
#                       vmin=0.0, vmax=None, xlabel='x', ylabel='y',  cax_label=r'$|\nabla\nu|$',
#                       fontsize=12)
# le.movie_plot_2D(glat.lattice.xy, glat.lattice.BL, 0 * glat.lattice.BL[:, 0],
#                  None, None, ax=plt.gca(), axcb=None, bondcolor='k',
#                  colorz=False, ptcolor=None, figsize='auto',
#                  colormap='BlueBlackRed', bgcolor='#ffffff', axis_off=False, axis_equal=True,
#                  lw=0.2)
# kpfns.add_kitaev_regions_to_plot(kcoll.cherns[glat_name][0], repi, offsetxy=np.array([15, 0]))
# plt.title(r'$|\nabla\nu|$')
# plt.ylabel(r'$y$')
# plt.xlabel(r'$x$')
# plt.savefig(outdir + '_nugrad.png')
#
# plt.clf()
# ind = 0
# leplt.plot_pcolormesh(xyvec[:, 0], xyvec[:, 1], nugrids_cut[repi].T.ravel(), 100,
#                       method='nearest', cmap='rwb0',
#                       vmin=-1.0, vmax=1.0, xlabel='x', ylabel='y',  cax_label=r'$\nu$',
#                       fontsize=12)
# le.movie_plot_2D(glat.lattice.xy, glat.lattice.BL, 0 * glat.lattice.BL[:, 0],
#                  None, None, ax=plt.gca(), axcb=None, bondcolor='k',
#                  colorz=False, ptcolor=None, figsize='auto',
#                  colormap='BlueBlackRed', bgcolor='#ffffff', axis_off=False, axis_equal=True,
#                  lw=0.2)
# kpfns.add_kitaev_regions_to_plot(kcoll.cherns[glat_name][0], repi, offsetxy=np.array([15, 0]))
# plt.title(r'$\nu$')
# plt.ylabel(r'$y$')
# plt.xlabel(r'$x$')
# plt.savefig(outdir + '_nugrid_cut.png')
#
# plt.clf()
# leplt.plot_pcolormesh(xyvec[:, 0], xyvec[:, 1], nugrad_cut[repi].T.ravel(), 100,
#                       method='nearest', cmap='viridis',
#                       vmin=0.0, vmax=None, xlabel='x', ylabel='y',  cax_label=r'$|\nabla\nu_c|$',
#                       fontsize=12)
# le.movie_plot_2D(glat.lattice.xy, glat.lattice.BL, 0 * glat.lattice.BL[:, 0],
#                  None, None, ax=plt.gca(), axcb=None, bondcolor='k',
#                  colorz=False, ptcolor=None, figsize='auto',
#                  colormap='BlueBlackRed', bgcolor='#ffffff', axis_off=False, axis_equal=True,
#                  lw=0.2)
# kpfns.add_kitaev_regions_to_plot(kcoll.cherns[glat_name][0], repi, offsetxy=np.array([15, 0]))
# plt.title(r'$|\nabla\nu_c|$')
# plt.ylabel(r'$y$')
# plt.xlabel(r'$x$')
# plt.savefig(outdir + '_nugrad_cut.png')
#
# plt.clf()
# leplt.plot_pcolormesh(xyvec[:, 0], xyvec[:, 1], nulap[repi].T.ravel(), 100,
#                       method='nearest', cmap='rwb0',
#                       vmin=None, vmax=None, xlabel='x', ylabel='y',  cax_label=r'$\nabla^2\nu$',
#                       fontsize=12)
# le.movie_plot_2D(glat.lattice.xy, glat.lattice.BL, 0 * glat.lattice.BL[:, 0],
#                  None, None, ax=plt.gca(), axcb=None, bondcolor='k',
#                  colorz=False, ptcolor=None, figsize='auto',
#                  colormap='BlueBlackRed', bgcolor='#ffffff', axis_off=False, axis_equal=True,
#                  lw=0.2)
# kpfns.add_kitaev_regions_to_plot(kcoll.cherns[glat_name][0], repi, offsetxy=np.array([15, 0]))
# plt.title(r'$\nabla^2\nu$')
# plt.ylabel(r'$y$')
# plt.xlabel(r'$x$')
# plt.savefig(outdir + '_nulaplace.png')
#
# plt.clf()
# leplt.plot_pcolormesh(xyvec[:, 0], xyvec[:, 1], np.abs(nulap[repi].T.ravel()), 100,
#                       method='nearest', cmap='viridis',
#                       vmin=None, vmax=None, xlabel='x', ylabel='y',  cax_label=r'$|\nabla^2\nu|$',
#                       fontsize=12)
# le.movie_plot_2D(glat.lattice.xy, glat.lattice.BL, 0 * glat.lattice.BL[:, 0],
#                  None, None, ax=plt.gca(), axcb=None, bondcolor='k',
#                  colorz=False, ptcolor=None, figsize='auto',
#                  colormap='BlueBlackRed', bgcolor='#ffffff', axis_off=False, axis_equal=True,
#                  lw=0.2)
# kpfns.add_kitaev_regions_to_plot(kcoll.cherns[glat_name][0], repi, offsetxy=np.array([15, 0]))
# plt.title(r'$|\nabla^2\nu|$')
# plt.ylabel(r'$y$')
# plt.xlabel(r'$x$')
# plt.savefig(outdir + '_nulaplaceabs.png')
#
# plt.clf()
# leplt.plot_pcolormesh(xyvec[:, 0], xyvec[:, 1], nugsum[repi].T.ravel(), 100,
#                       method='nearest', cmap='viridis',
#                       vmin=None, vmax=None, xlabel='x', ylabel='y',  cax_label=r'$\sum_j |\nabla_j \nu|$',
#                       fontsize=12)
# le.movie_plot_2D(glat.lattice.xy, glat.lattice.BL, 0 * glat.lattice.BL[:, 0],
#                  None, None, ax=plt.gca(), axcb=None, bondcolor='k',
#                  colorz=False, ptcolor=None, figsize='auto',
#                  colormap='BlueBlackRed', bgcolor='#ffffff', axis_off=False, axis_equal=True,
#                  lw=0.2)
# kpfns.add_kitaev_regions_to_plot(kcoll.cherns[glat_name][0], repi2, offsetxy=np.array([15, 0]))
# plt.title(r'$|\sum_j \nabla_j \nu|$')
# plt.ylabel(r'$y$')
# plt.xlabel(r'$x$')
# plt.savefig(outdir + '_nugradsumabs.png')
#
# #######################################################
# #######################################################
# # LOOK AT REPI2
# plt.clf()
# ind = 0
# leplt.plot_pcolormesh(xyvec[:, 0], xyvec[:, 1], nugrids[repi2].T.ravel(), 100,
#                       method='nearest', cmap='rwb0',
#                       vmin=-1.0, vmax=1.0, xlabel='x', ylabel='y',  cax_label=r'$\nu$',
#                       fontsize=12)
# le.movie_plot_2D(glat.lattice.xy, glat.lattice.BL, 0 * glat.lattice.BL[:, 0],
#                  None, None, ax=plt.gca(), axcb=None, bondcolor='k',
#                  colorz=False, ptcolor=None, figsize='auto',
#                  colormap='BlueBlackRed', bgcolor='#ffffff', axis_off=False, axis_equal=True,
#                  lw=0.2)
# kpfns.add_kitaev_regions_to_plot(kcoll.cherns[glat_name][0], repi2, offsetxy=np.array([15, 0]))
# plt.title(r'$\nu$')
# plt.ylabel(r'$y$')
# plt.xlabel(r'$x$')
# plt.savefig(outdir + '_nugrid_early.png')
#
# plt.clf()
# leplt.plot_pcolormesh(xyvec[:, 0], xyvec[:, 1], nugrad[repi2].T.ravel(), 100,
#                       method='nearest', cmap='viridis',
#                       vmin=0.0, vmax=None, xlabel='x', ylabel='y',  cax_label=r'$|\nabla\nu|$',
#                       fontsize=12)
# le.movie_plot_2D(glat.lattice.xy, glat.lattice.BL, 0 * glat.lattice.BL[:, 0],
#                  None, None, ax=plt.gca(), axcb=None, bondcolor='k',
#                  colorz=False, ptcolor=None, figsize='auto',
#                  colormap='BlueBlackRed', bgcolor='#ffffff', axis_off=False, axis_equal=True,
#                  lw=0.2)
# kpfns.add_kitaev_regions_to_plot(kcoll.cherns[glat_name][0], repi2, offsetxy=np.array([15, 0]))
# plt.title(r'$|\nabla\nu|$')
# plt.ylabel(r'$y$')
# plt.xlabel(r'$x$')
# plt.savefig(outdir + '_nugrad_early.png')
#
# plt.clf()
# leplt.plot_pcolormesh(xyvec[:, 0], xyvec[:, 1], nugrad_cut[repi2].T.ravel(), 100,
#                       method='nearest', cmap='viridis',
#                       vmin=0.0, vmax=None, xlabel='x', ylabel='y',  cax_label=r'$|\nabla\nu|$',
#                       fontsize=12)
# le.movie_plot_2D(glat.lattice.xy, glat.lattice.BL, 0 * glat.lattice.BL[:, 0],
#                  None, None, ax=plt.gca(), axcb=None, bondcolor='k',
#                  colorz=False, ptcolor=None, figsize='auto',
#                  colormap='BlueBlackRed', bgcolor='#ffffff', axis_off=False, axis_equal=True,
#                  lw=0.2)
# kpfns.add_kitaev_regions_to_plot(kcoll.cherns[glat_name][0], repi2, offsetxy=np.array([15, 0]))
# plt.title(r'$|\nabla\nu|$')
# plt.ylabel(r'$y$')
# plt.xlabel(r'$x$')
# plt.savefig(outdir + '_nugrad_cut_early.png')
#
# plt.clf()
# leplt.plot_pcolormesh(xyvec[:, 0], xyvec[:, 1], nulap[repi2].T.ravel(), 100,
#                       method='nearest', cmap='rwb0',
#                       vmin=None, vmax=None, xlabel='x', ylabel='y',  cax_label=r'$|\nabla^2\nu|$',
#                       fontsize=12)
# le.movie_plot_2D(glat.lattice.xy, glat.lattice.BL, 0 * glat.lattice.BL[:, 0],
#                  None, None, ax=plt.gca(), axcb=None, bondcolor='k',
#                  colorz=False, ptcolor=None, figsize='auto',
#                  colormap='BlueBlackRed', bgcolor='#ffffff', axis_off=False, axis_equal=True,
#                  lw=0.2)
# kpfns.add_kitaev_regions_to_plot(kcoll.cherns[glat_name][0], repi2, offsetxy=np.array([15, 0]))
# plt.title(r'$|\nabla^2\nu|$')
# plt.ylabel(r'$y$')
# plt.xlabel(r'$x$')
# plt.savefig(outdir + '_nulaplace_early.png')
#
# plt.clf()
# leplt.plot_pcolormesh(xyvec[:, 0], xyvec[:, 1], nugsum[repi2].T.ravel(), 100,
#                       method='nearest', cmap='viridis',
#                       vmin=None, vmax=None, xlabel='x', ylabel='y',  cax_label=r'$|\sum_j \nabla_j \nu|$',
#                       fontsize=12)
# le.movie_plot_2D(glat.lattice.xy, glat.lattice.BL, 0 * glat.lattice.BL[:, 0],
#                  None, None, ax=plt.gca(), axcb=None, bondcolor='k',
#                  colorz=False, ptcolor=None, figsize='auto',
#                  colormap='BlueBlackRed', bgcolor='#ffffff', axis_off=False, axis_equal=True,
#                  lw=0.2)
# kpfns.add_kitaev_regions_to_plot(kcoll.cherns[glat_name][0], repi2, offsetxy=np.array([15, 0]))
# plt.title(r'$|\sum_j \nabla_j \nu|$')
# plt.ylabel(r'$y$')
# plt.xlabel(r'$x$')
# plt.savefig(outdir + '_nugradsumabs_early.png')
#
# ##########################################
# ##########################################
# plt.clf()
# ind = 0
# for ll in local:
#     plt.plot(np.abs(np.imag(eigval[top])), ll, '-', color=copperMap.to_rgba(ind))
#     ind += 1
# plt.title(r'Edge localization $ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 |\nabla\nu|_i$')
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$\omega$')
# plt.savefig(outdir + '_gradnu_vs_omega.png')
#
# plt.clf()
# ind = 0
# for ll in local2:
#     plt.plot(np.abs(np.imag(eigval[top])), ll, '-', color=copperMap.to_rgba(ind))
#     ind += 1
# plt.title(r'Edge localization $ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 \nabla^2\nu_i$')
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$\omega$')
# plt.savefig(outdir + '_laplacenu_vs_omega.png')
#
# plt.clf()
# ind = 0
# for ll in local4:
#     plt.plot(np.abs(np.imag(eigval[top])), ll, '-', color=copperMap.to_rgba(ind))
#     ind += 1
# plt.title(r'Edge localization $ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 |\nabla^2\nu|_i$')
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$\omega$')
# plt.savefig(outdir + '_laplacenuabs_vs_omega.png')
#
# plt.clf()
# ind = 0
# for ll in local5:
#     plt.plot(np.abs(np.imag(eigval[top])), ll, '-', color=copperMap.to_rgba(ind))
#     ind += 1
# plt.title(r'Edge localization $ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 \sum_j |\partial_j \nu|_i$')
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$\omega$')
# plt.savefig(outdir + '_gradsumnuabs_vs_omega.png')
#
# plt.clf()
# ind = 0
# for ll in local6:
#     plt.plot(np.abs(np.imag(eigval[top])), ll, '-', color=copperMap.to_rgba(ind))
#     ind += 1
# plt.title(r'Edge localization smoothed $ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 \sum_j |\partial_j \nu|_i$')
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$\omega$')
# plt.savefig(outdir + '_gradsumnuabssmooth_vs_omega.png')
#
# # Correlate with ipr
# if glat.ipr is None:
#     ipr = glat.get_ipr()
#
# plt.clf()
# ind = 0
# for ll in local:
#     plt.scatter(ipr[top], ll, color=copperMap.to_rgba(ind))
#     ind += 1
# plt.title(r'Edge localization vs $p$, $ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 |\nabla\nu|_i$')
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$p^{-1}$')
# plt.savefig(outdir + '_gradnu_vs_ipr.png')
#
# plt.clf()
# ind = 0
# for ll in local:
#     plt.scatter(1./ipr[top], ll, s=20, color=copperMap.to_rgba(ind))
#     ind += 1
# plt.title(r'Edge localization vs $p$, $ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 |\nabla\nu|_i$')
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$p$')
# plt.savefig(outdir + '_gradnu_vs_p.png')
#
# plt.clf()
# ind = 0
# for ind in range(len(eigval)/2):
#     sm = plt.scatter(1./ipr[ind], local[repi, ind], s=sz, color=jetMap.to_rgba(np.abs(eigval[ind])))
#     ind += 1
# sm = plt.cm.ScalarMappable(cmap=jet, norm=plt.Normalize(vmin=1.0, vmax=max(np.imag(eigval))))
# # fake up the array of the scalar mappable.
# sm._A = []
#
# plt.colorbar(sm, label=r'$\omega$')
# plt.title(r'Edge localization vs $p$, $ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 |\nabla\nu|_i$')
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$p$')
# plt.savefig(outdir + '_gradnu_vsoromega.png')
#
# plt.clf()
# ind = 0
# for ind in range(len(eigval)/2):
#     sm = plt.scatter(ipr[ind], local[repi, ind], s=sz, color=jetMap.to_rgba(np.abs(eigval[ind])))
#     ind += 1
# sm = plt.cm.ScalarMappable(cmap=jet, norm=plt.Normalize(vmin=1.0, vmax=max(np.imag(eigval))))
# # fake up the array of the scalar mappable.
# sm._A = []
# plt.colorbar(sm, label=r'$\omega$')
# plt.title(r'Edge localization vs $p$, $ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 |\nabla\nu|_i$')
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$p^{-1}$')
# plt.savefig(outdir + '_gradnu_vs_ipr_coloromega.png')
# plt.xlim(0, 30)
# plt.savefig(outdir + '_gradnu_vs_ipr_coloromega_zoom.png')
#
# plt.clf()
# ind = 0
# for ind in range(len(eigval)/2):
#     sm = plt.scatter(ipr[ind], local4[repi, ind], s=sz, color=jetMap.to_rgba(np.abs(eigval[ind])))
#     ind += 1
# sm = plt.cm.ScalarMappable(cmap=jet, norm=plt.Normalize(vmin=1.0, vmax=max(np.imag(eigval))))
# # fake up the array of the scalar mappable.
# sm._A = []
# plt.colorbar(sm, label=r'$\omega$')
# plt.title(r'Edge localization vs $p$, $ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 |\nabla^2\nu|_i$')
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$p^{-1}$')
# plt.savefig(outdir + '_laplacenuabs_vs_ipr_coloromega.png')
# plt.xlim(0, 30)
# plt.savefig(outdir + '_laplacenuabs_vs_ipr_coloromega_zoom.png')
#
# plt.clf()
# ind = 0
# for ind in range(len(eigval)/2):
#     sm = plt.scatter(ipr[ind], local5[repi, ind], s=sz, color=jetMap.to_rgba(np.abs(eigval[ind])))
#     ind += 1
# sm = plt.cm.ScalarMappable(cmap=jet, norm=plt.Normalize(vmin=1.0, vmax=max(np.imag(eigval))))
# # fake up the array of the scalar mappable.
# sm._A = []
# plt.colorbar(sm, label=r'$\omega$')
# ttl = r'Edge localization vs $p$, $ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 \sum_j |\partial_j \nu|_i$'
# plt.title(ttl)
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$p^{-1}$')
# plt.savefig(outdir + '_gradsumnuabs_vs_ipr_coloromega.png')
# plt.xlim(0, 30)
# plt.savefig(outdir + '_gradsumnuabs_vs_ipr_coloromega_zoom.png')
#
# plt.clf()
# ind = 0
# for ind in range(len(eigval)/2):
#     sm = plt.scatter(1./ipr[ind], local5[repi, ind], s=sz, color=jetMap.to_rgba(np.abs(eigval[ind])))
#     ind += 1
# sm = plt.cm.ScalarMappable(cmap=jet, norm=plt.Normalize(vmin=1.0, vmax=max(np.imag(eigval))))
# # fake up the array of the scalar mappable.
# sm._A = []
# plt.colorbar(sm, label=r'$\omega$')
# ttl = r'Edge localization vs $p$, $ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 \sum_j |\partial_j \nu|_i$'
# plt.title(ttl)
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$p$')
# plt.savefig(outdir + '_gradsumnuabs_vs_p_coloromega.png')
#
# plt.clf()
# ind = 0
# for ind in range(len(eigval)/2):
#     sm = plt.scatter(1./ipr[ind], local6[repi, ind], s=sz, color=jetMap.to_rgba(np.abs(eigval[ind])))
#     ind += 1
# sm = plt.cm.ScalarMappable(cmap=jet, norm=plt.Normalize(vmin=1.0, vmax=max(np.imag(eigval))))
# # fake up the array of the scalar mappable.
# sm._A = []
# plt.colorbar(sm, label=r'$\omega$')
# ttl = r'Edge localization vs $p$, smoothed $ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 \sum_j |\partial_j \nu|_i$'
# plt.title(ttl)
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$p$')
# plt.savefig(outdir + '_gradsumnuabssmooth_vs_p_coloromega.png')
#
#
# ####################################
# # REPI
# ####################################
# plt.clf()
# ind = 0
# for ind in range(len(eigval)/2):
#     sm = plt.scatter(abs(eigval[ind]) - 2.25, local[repi, ind], s=sz,
#                      color=jetMap.to_rgba(np.abs(eigval[ind])))
#     ind += 1
# sm = plt.cm.ScalarMappable(cmap=jet, norm=plt.Normalize(vmin=1.0, vmax=max(np.imag(eigval))))
# # fake up the array of the scalar mappable.
# sm._A = []
# plt.colorbar(sm, label=r'$\omega$')
# plt.title(r'Edge localization vs $\omega-\omega_c$, ' +
#           r'$ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 |\nabla\nu|_i$')
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$\omega - 2.25 \Omega_g$')
# plt.savefig(outdir + '_gradnu_vs_omegadiff_coloromega.png')
#
# plt.clf()
# ind = 0
# for ind in range(len(eigval)/2):
#     sm = plt.scatter(abs(eigval[ind]) - 2.25, local2[repi, ind], s=sz,
#                      color=jetMap.to_rgba(np.abs(eigval[ind])))
#     ind += 1
# sm = plt.cm.ScalarMappable(cmap=jet, norm=plt.Normalize(vmin=1.0, vmax=max(np.imag(eigval))))
# # fake up the array of the scalar mappable.
# sm._A = []
#
# plt.colorbar(sm, label=r'$\omega$')
# plt.title(r'Edge localization vs $\omega-\omega_c$, ' +
#           r'$ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 \nabla^2\nu_i$')
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$\omega - 2.25 \Omega_g$')
# plt.savefig(outdir + '_laplacenu_vs_omegadiff_coloromega.png')
#
# plt.clf()
# ind = 0
# for ind in range(len(eigval)/2):
#     sm = plt.scatter(abs(eigval[ind]) - 2.25, local3[repi, ind], s=sz,
#                      color=jetMap.to_rgba(np.abs(eigval[ind])))
#     ind += 1
# sm = plt.cm.ScalarMappable(cmap=jet, norm=plt.Normalize(vmin=1.0, vmax=max(np.imag(eigval))))
# sm._A = []
# plt.colorbar(sm, label=r'$\omega$')
# plt.title(r'Edge localization vs $\omega-\omega_c$, ' +
#           r'$ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 |\nabla\nu_c|_i$')
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$\omega - 2.25 \Omega_g$')
# plt.savefig(outdir + '_gcnu_vs_omegadiff_coloromega.png')
#
# plt.clf()
# ind = 0
# for ind in range(len(eigval)/2):
#     sm = plt.scatter(abs(eigval[ind]) - 2.25, local4[repi, ind], s=sz,
#                      color=jetMap.to_rgba(np.abs(eigval[ind])))
#     ind += 1
# sm = plt.cm.ScalarMappable(cmap=jet, norm=plt.Normalize(vmin=1.0, vmax=max(np.imag(eigval))))
# # fake up the array of the scalar mappable.
# sm._A = []
# plt.colorbar(sm, label=r'$\omega$')
# plt.title(r'Edge localization vs $\omega-\omega_c$, ' +
#           r'$ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 |\nabla^2\nu|_i$')
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$\omega - 2.25 \Omega_g$')
# plt.savefig(outdir + '_laplacenuabs_vs_omegadiff_coloromega.png')
#
# plt.clf()
# ind = 0
# for ind in range(len(eigval)/2):
#     sm = plt.scatter(abs(eigval[ind]) - 2.25, local5[repi, ind], s=sz,
#                      color=jetMap.to_rgba(np.abs(eigval[ind])))
#     ind += 1
# sm = plt.cm.ScalarMappable(cmap=jet, norm=plt.Normalize(vmin=1.0, vmax=max(np.imag(eigval))))
# # fake up the array of the scalar mappable.
# sm._A = []
# plt.colorbar(sm, label=r'$\omega$')
# plt.title(r'Edge localization vs $\omega-\omega_c$, ' +
#           r'$ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 \sum_j |\partial_j \nu|_i$')
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$\omega - 2.25 \Omega_g$')
# plt.savefig(outdir + '_gradsumnuabs_vs_omegadiff_coloromega.png')
#
# plt.clf()
# ind = 0
# for ind in range(len(eigval)/2):
#     sm = plt.scatter(abs(eigval[ind]) - 2.25, local5[repi, ind]*ipr[ind], s=sz,
#                      color=jetMap.to_rgba(np.abs(eigval[ind])))
#     ind += 1
# sm = plt.cm.ScalarMappable(cmap=jet, norm=plt.Normalize(vmin=1.0, vmax=max(np.imag(eigval))))
# sm._A = []
# plt.colorbar(sm, label=r'$\omega$')
# plt.title(r'Edge localization vs $\omega-\omega_c$, ' +
#           r'$ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 \sum_j |\partial_j \nu|_i$')
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$\omega - 2.25 \Omega_g$')
# plt.savefig(outdir + '_gradsumnuabsipr_vs_omegadiff_coloromega.png')
# plt.ylim(0, 5)
# plt.savefig(outdir + '_gradsumnuabsipr_vs_omegadiff_coloromega_zoom.png')
#
# plt.clf()
# ind = 0
# for ind in range(len(eigval)/2):
#     sm = plt.scatter(abs(eigval[ind]), local6[repi, ind], s=sz,
#                      color=jetMap.to_rgba(np.abs(eigval[ind])))
#     ind += 1
# sm = plt.cm.ScalarMappable(cmap=jet, norm=plt.Normalize(vmin=1.0, vmax=max(np.imag(eigval))))
# sm._A = []
# plt.colorbar(sm, label=r'$\omega$')
# plt.title(r'Edge localization vs $\omega-\omega_c$, ' +
#           r'$ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 \sum_j |\partial_j \nu|_i$')
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$\omega$')
# plt.savefig(outdir + '_gradsumnuabssmooth_vs_omega_coloromega.png')
#
# plt.clf()
# ind = 0
# for ind in range(len(eigval)/2):
#     sm = plt.scatter(abs(abs(eigval[ind]) - 2.25), local[repi, ind], s=sz,
#                      color=jetMap.to_rgba(np.abs(eigval[ind])))
#     ind += 1
# sm = plt.cm.ScalarMappable(cmap=jet, norm=plt.Normalize(vmin=1.0, vmax=max(np.imag(eigval))))
# sm._A = []
# plt.colorbar(sm, label=r'$\omega$')
# plt.title(r'Edge localization vs $\omega-\omega_c$, ' +
#           r'$ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 |\nabla\nu|_i$')
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$|\omega - 2.25 \Omega_g|$')
# plt.savefig(outdir + '_gradnu_vs_absomegadiff_coloromega.png')
#
# ####################################
# # REPI2
# ####################################
# plt.clf()
# ind = 0
# for ind in range(len(eigval)/2):
#     sm = plt.scatter(abs(eigval[ind]) - 2.25, local[repi2, ind], s=sz,
#                      color=jetMap.to_rgba(np.abs(eigval[ind])))
#     ind += 1
# sm = plt.cm.ScalarMappable(cmap=jet, norm=plt.Normalize(vmin=1.0, vmax=max(np.imag(eigval))))
# sm._A = []
# plt.colorbar(sm, label=r'$\omega$')
# plt.title(r'Edge localization vs $\omega-\omega_c$, ' +
#           r'$ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 |\nabla\nu|_i$')
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$\omega - 2.25 \Omega_g$')
# plt.savefig(outdir + '_gradnu_vs_omegadiff_coloromega_early.png')
#
# plt.clf()
# ind = 0
# for ind in range(len(eigval)/2):
#     sm = plt.scatter(abs(eigval[ind]) - 2.25, local2[repi2, ind], s=sz,
#                      color=jetMap.to_rgba(np.abs(eigval[ind])))
#     ind += 1
# sm = plt.cm.ScalarMappable(cmap=jet, norm=plt.Normalize(vmin=1.0, vmax=max(np.imag(eigval))))
# sm._A = []
# plt.colorbar(sm, label=r'$\omega$')
# plt.title(r'Edge localization vs $\omega-\omega_c$, ' +
#           r'$ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 \nabla^2\nu_i$')
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$\omega - 2.25 \Omega_g$')
# plt.savefig(outdir + '_laplacenu_vs_omegadiff_coloromega_early.png')
#
# plt.clf()
# ind = 0
# for ind in range(len(eigval)/2):
#     sm = plt.scatter(abs(eigval[ind]) - 2.25, local3[repi2, ind], s=sz,
#                      color=jetMap.to_rgba(np.abs(eigval[ind])))
#     ind += 1
# sm = plt.cm.ScalarMappable(cmap=jet, norm=plt.Normalize(vmin=1.0, vmax=max(np.imag(eigval))))
# sm._A = []
# plt.colorbar(sm, label=r'$\omega$')
# plt.title(r'Edge localization vs $\omega-\omega_c$, ' +
#           r'$ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 |\nabla\nu_c|_i$')
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$\omega - 2.25 \Omega_g$')
# plt.savefig(outdir + '_gcnu_vs_omegadiff_coloromega_early.png')
#
# plt.clf()
# ind = 0
# for ind in range(len(eigval)/2):
#     sm = plt.scatter(abs(eigval[ind]) - 2.25, local4[repi2, ind], s=sz,
#                      color=jetMap.to_rgba(np.abs(eigval[ind])))
#     ind += 1
# sm = plt.cm.ScalarMappable(cmap=jet, norm=plt.Normalize(vmin=1.0, vmax=max(np.imag(eigval))))
# sm._A = []
# plt.colorbar(sm, label=r'$\omega$')
# plt.title(r'Edge localization vs $\omega-\omega_c$, ' +
#           r'$ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 |\nabla^2\nu|_i$')
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$\omega - 2.25 \Omega_g$')
# plt.savefig(outdir + '_laplacenuabs_vs_omegadiff_coloromega_early.png')
#
# plt.clf()
# ind = 0
# for ind in range(len(eigval)/2):
#     sm = plt.scatter(abs(eigval[ind]) - 2.25, local5[repi2, ind], s=sz,
#                      color=jetMap.to_rgba(np.abs(eigval[ind])))
#     ind += 1
# sm = plt.cm.ScalarMappable(cmap=jet, norm=plt.Normalize(vmin=1.0, vmax=max(np.imag(eigval))))
# sm._A = []
# plt.colorbar(sm, label=r'$\omega$')
# plt.title(r'Edge localization vs $\omega-\omega_c$, ' +
#           r'$ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 \sum_j |\partial_j \nu|_i$')
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$\omega - 2.25 \Omega_g$')
# plt.savefig(outdir + '_gradsumnuabs_vs_omegadiff_coloromega_early.png')
#
# plt.clf()
# ind = 0
# for ind in range(len(eigval)/2):
#     sm = plt.scatter(abs(eigval[ind]) - 2.25, local6[repi2, ind], s=sz,
#                      color=jetMap.to_rgba(np.abs(eigval[ind])))
#     ind += 1
# sm = plt.cm.ScalarMappable(cmap=jet, norm=plt.Normalize(vmin=1.0, vmax=max(np.imag(eigval))))
# sm._A = []
# plt.colorbar(sm, label=r'$\omega$')
# plt.title(r'Edge localization vs $\omega-\omega_c$, smoothed' +
#           r'$ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 \sum_j |\partial_j \nu|_i$')
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$\omega - 2.25 \Omega_g$')
# plt.savefig(outdir + '_gradsumnuabssmooth_vs_omegadiff_coloromega_early.png')
#
# plt.clf()
# ind = 0
# for ind in range(len(eigval)/2):
#     sm = plt.scatter(abs(abs(eigval[ind]) - 2.25), local5[repi2, ind], s=sz,
#                      color=jetMap.to_rgba(np.abs(eigval[ind])))
#     ind += 1
# sm = plt.cm.ScalarMappable(cmap=jet, norm=plt.Normalize(vmin=1.0, vmax=max(np.imag(eigval))))
# sm._A = []
# plt.colorbar(sm, label=r'$\omega$')
# plt.title(r'Edge localization vs $\omega-\omega_c$, ' +
#           r'$ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 |\nabla\nu|_i$')
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$|\omega - 2.25 \Omega_g|$')
# plt.savefig(outdir + '_gradnusumabs_vs_absomegadiff_coloromega_early.png')
#
# ######################################################
# # AVERAGING grads
# plt.clf()
# ind = 0
# inds = range(len(eigval)/2)
# evrm3 = le.running_mean(eigval[inds], 3)
# for rii in range(0, len(local5)-3, int(len(local5)/20.0)):
#     l5rm3 = le.running_mean(local5[rii, inds], 3)
#     sm = plt.plot(abs(evrm3) - 2.25, l5rm3, '-', color=copperMap.to_rgba(rii/(len(local5)-3)))
# sm = plt.cm.ScalarMappable(cmap=copper, norm=plt.Normalize(vmin=0.0, vmax=1.0))
# sm._A = []
# plt.colorbar(sm, label=r'kitaev region size')
# plt.title(r'Edge localization vs $\omega-\omega_c$, ' +
#           r'$ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 \sum_j |\partial_j \nu|_i$')
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$\omega - 2.25 \Omega_g$')
# plt.savefig(outdir + '_gradsumnuabs_vs_omegadiff_coloromega_avg3.png')
#
# plt.clf()
# ind = 0
# inds = range(len(eigval)/2)
# evrm5 = le.running_mean(eigval[inds], 5)
# for rii in range(0, len(local5)-5, int(len(local5)/20.0)):
#     l5rm5 = le.running_mean(local5[rii, inds], 5)
#     sm = plt.plot(abs(evrm5) - 2.25, l5rm5, '-', color=copperMap.to_rgba(rii/(len(local5)-5)))
#     ind += 1
# sm = plt.cm.ScalarMappable(cmap=copper, norm=plt.Normalize(vmin=0.0, vmax=1.0))
# sm._A = []
# plt.colorbar(sm, label=r'$\omega$')
# plt.title(r'Edge localization vs $\omega-\omega_c$, ' +
#           r'$ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 \sum_j |\partial_j \nu|_i$')
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$\omega - 2.25 \Omega_g$')
# plt.savefig(outdir + '_gradsumnuabs_vs_omegadiff_coloromega_avg5.png')
#
# plt.clf()
# ind = 0
# inds = range(len(eigval)/2)
# evrm10 = le.running_mean(eigval[inds], 10)
# for rii in range(0, len(local5)-10, int(len(local5)/20.0)):
#     print 'rii = ', rii
#     l5rm10 = le.running_mean(local5[rii, inds], 10)
#     sm = plt.plot(abs(evrm10) - 2.25, l5rm10, '-', color=copperMap.to_rgba(rii/(len(local5)-10)))
# sm = plt.cm.ScalarMappable(cmap=copper, norm=plt.Normalize(vmin=0.0, vmax=1.0))
# sm._A = []
# plt.colorbar(sm, label=r'$\omega$')
# plt.title(r'Edge localization vs $\omega-\omega_c$, ' +
#           r'$ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 \sum_j |\partial_j \nu|_i$')
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$\omega - 2.25 \Omega_g$')
# plt.savefig(outdir + '_gradsumnuabs_vs_omegadiff_coloromega_avg10.png')
#
# plt.clf()
# ind = 0
# inds = range(len(eigval)/2)
# evrm10 = le.running_mean(eigval[inds], 10)
# for rii in range(0, len(local5)-10, int(len(local6)/20.0)):
#     print 'rii = ', rii
#     l5rm10 = le.running_mean(local5[rii, inds], 10)
#     sm = plt.plot(abs(evrm10) - 2.25, l5rm10, '-', color=copperMap.to_rgba(rii/(len(local5)-10)))
# sm = plt.cm.ScalarMappable(cmap=copper, norm=plt.Normalize(vmin=0.0, vmax=1.0))
# sm._A = []
# plt.colorbar(sm, label=r'$\omega$')
# plt.title(r'Edge localization vs $\omega-\omega_c$, smoothed ' +
#           r'$ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 \sum_j |\partial_j \nu|_i$')
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$\omega - 2.25 \Omega_g$')
# plt.savefig(outdir + '_gradsumnuabssmooth_vs_omegadiff_coloromega_avg10.png')
#
# plt.clf()
# ind = 0
# inds = range(len(eigval)/2)
# evrm10 = le.running_mean(eigval[inds], 20)
# for rii in range(0, len(local5)-20, int(len(local5)/20.0)):
#     print 'rii = ', rii
#     l5rm20 = le.running_mean(local5[rii, inds], 20)
#     sm = plt.plot(abs(evrm10) - 2.25, l5rm20, '-', color=copperMap.to_rgba(rii/(len(local5)-20)))
# sm = plt.cm.ScalarMappable(cmap=copper, norm=plt.Normalize(vmin=0.0, vmax=1.0))
# sm._A = []
# plt.colorbar(sm, label=r'$\omega$')
# plt.title(r'Edge localization vs $\omega-\omega_c$, ' +
#           r'$ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 \sum_j |\partial_j \nu|_i$')
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$\omega - 2.25 \Omega_g$')
# plt.savefig(outdir + '_gradsumnuabs_vs_omegadiff_coloromega_avg20.png')
#
# ######################################################
# # AVERAGING Laplacian
# plt.clf()
# ind = 0
# inds = range(len(eigval)/2)
# evrm3 = le.running_mean(eigval[inds], 3)
# for rii in range(0, len(local5)-3, int(len(local4)/20.0)):
#     l4rm3 = le.running_mean(np.abs(local4[rii, inds]), 3)
#     sm = plt.plot(abs(evrm3) - 2.25, l4rm3, '-', color=copperMap.to_rgba(rii/(len(local5)-3)))
# sm = plt.cm.ScalarMappable(cmap=copper, norm=plt.Normalize(vmin=0.0, vmax=1.0))
# sm._A = []
# plt.colorbar(sm, label=r'$\omega$')
# plt.title(r'Edge localization vs $\omega-\omega_c$, ' +
#           r'$ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 \sum_j |\partial_j \nu|_i$')
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$\omega - 2.25 \Omega_g$')
# plt.savefig(outdir + '_laplacenuabs_vs_omegadiff_coloromega_avg3.png')
#
# plt.clf()
# ind = 0
# inds = range(len(eigval)/2)
# evrm5 = le.running_mean(eigval[inds], 5)
# for rii in range(0, len(local5)-5, int(len(local4)/20.0)):
#     l4rm5 = le.running_mean(np.abs(local4[rii, inds]), 5)
#     sm = plt.plot(abs(evrm5) - 2.25, l4rm5, '-', color=copperMap.to_rgba(rii/(len(local5)-5)))
#     ind += 1
# sm = plt.cm.ScalarMappable(cmap=copper, norm=plt.Normalize(vmin=0.0, vmax=1.0))
# sm._A = []
# plt.colorbar(sm, label=r'$\omega$')
# plt.title(r'Edge localization vs $\omega-\omega_c$, ' +
#           r'$ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 \sum_j |\partial_j \nu|_i$')
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$\omega - 2.25 \Omega_g$')
# plt.savefig(outdir + '_laplacenuabs_vs_omegadiff_coloromega_avg5.png')
#
# plt.clf()
# ind = 0
# inds = range(len(eigval)/2)
# evrm10 = le.running_mean(eigval[inds], 10)
# for rii in range(0, len(local5)-10, int(len(local4)/20.0)):
#     l4rm10 = le.running_mean(np.abs(local4[rii, inds]), 10)
#     sm = plt.plot(abs(evrm10) - 2.25, l4rm10, '-', color=copperMap.to_rgba(rii/(len(local5)-10)))
# sm = plt.cm.ScalarMappable(cmap=copper, norm=plt.Normalize(vmin=0.0, vmax=1.0))
# sm._A = []
# plt.colorbar(sm, label=r'$\omega$')
# plt.title(r'Edge localization vs $\omega-\omega_c$, ' +
#           r'$ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 \sum_j |\partial_j \nu|_i$')
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$\omega - 2.25 \Omega_g$')
# plt.savefig(outdir + '_laplacenuabs_vs_omegadiff_coloromega_avg10.png')
#
# plt.clf()
# ind = 0
# inds = range(len(eigval)/2)
# evrm10 = le.running_mean(eigval[inds], 20)
# for rii in range(0, len(local5)-20, int(len(local5)/20.0)):
#     l4rm20 = le.running_mean(np.abs(local4[rii, inds]), 20)
#     sm = plt.plot(abs(evrm10) - 2.25, l4rm20, '-', color=copperMap.to_rgba(rii/(len(local5)-3)))
# sm = plt.cm.ScalarMappable(cmap=copper, norm=plt.Normalize(vmin=0.0, vmax=1.0))
# sm._A = []
# plt.colorbar(sm, label=r'$\omega$')
# plt.title(r'Edge localization vs $\omega-\omega_c$, ' +
#           r'$ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 \sum_j |\partial_j \nu|_i$')
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$\omega - 2.25 \Omega_g$')
# plt.savefig(outdir + '_laplacenuabs_vs_omegadiff_coloromega_avg20.png')
#
# ######################################################
# # MEDIAN grads
# plt.clf()
# step = 0.1
# bins = np.arange(1.0, 4.0, step)
# digits = np.digitize(np.abs(eigval[inds]), bins, right=False)
# ind = 0
# for ind in range(len(bins)-1):
#     inds[ind] = np.where(digits == ind)[0]
#
# for rii in range(0, len(local5), int(len(local5)/20.0)):
#     medloc5[ind] = np.median(local5[rii, inds])
#     plt.plot(bins[0:len(bins)-1], medloc5[rii], '-', color=copperMap.to_rgba(rii/(len(local5))))
# sm = plt.cm.ScalarMappable(cmap=copper, norm=plt.Normalize(vmin=5.0, vmax=1.0))
# sm._A = []
# plt.colorbar(sm, label=r'$kitaev region size (arb units)$')
# plt.title(r'Edge localization vs $\omega-\omega_c$, ' +
#           r'$ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 \sum_j |\partial_j \nu|_i$')
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$\omega - 2.25 \Omega_g$')
# plt.savefig(outdir + '_gradsumnuabs_vs_omegad_coloromega_median0p1.png')
#
# medloc6 = np.zeros((len(local6), len(bins)-1), dtype=float)
# for ind in range(len(bins)-1):
#     inds[ind] = np.where(digits == ind)[0]
#     medloc6[ind] = np.median(local6[inds])
#
# for rii in range(0, len(local6), int(len(local6)/20.0)):
#     medloc5[ind] = np.median(local5[rii, inds])
#     plt.plot(bins[0:len(bins)-1], medloc6[rii], '-', color=copperMap.to_rgba(rii/(len(local6))))
# sm = plt.cm.ScalarMappable(cmap=copper, norm=plt.Normalize(vmin=5.0, vmax=1.0))
# sm._A = []
# plt.colorbar(sm, label=r'$kitaev region size (arb units)$')
# plt.title(r'Edge localization vs $\omega-\omega_c$, ' +
#           r'$ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 \sum_j |\partial_j \nu|_i$')
# plt.ylabel(r'$\zeta$')
# plt.xlabel(r'$\omega - 2.25 \Omega_g$')
# plt.savefig(outdir + '_gradsumnuabssmooth_vs_omegad_coloromega_median0p1.png')
