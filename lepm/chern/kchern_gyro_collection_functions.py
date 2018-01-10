import lepm.kitaev.kitaev_collection_functions as kcollfns

''''''


def gap_midpoints_honeycomb(delta):
    """

    Parameters
    ----------
    delta

    Returns
    -------

    """
    return kcollfns.gap_midpoints_honeycomb(delta)


def retrieve_param_value(lp_value):
    """

    Parameters
    ----------
    lp_value

    Returns
    -------

    """
    return kcollfns.retrieve_param_value(lp_value)


def plot_cherns_vary_param(kcgcoll, param_type='glat', sz_param_nu=None,
                           reverse=False, param='percolation_density',
                           title='Chern index calculation', xlabel=None):
    """

    Parameters
    ----------
    kcgcoll :
    param_type :
    sz_param_nu :
    reverse :
    param :
    title :
    xlabel :

    Returns
    -------
    """
    if sz_param_nu is None:
        if param_type == 'lat' or param_type == 'lp':
            param_nu = self.collect_cherns_vary_lpparam(param=param, reverse=reverse)
        elif param_type == 'glat':
            param_nu = self.collect_cherns_vary_glatparam(param=param, reverse=reverse)
        else:
            raise RuntimeError("param_type argument passed is not 'glat' or 'lat/lp'")

    # Plot it as colormap
    plt.close('all')

    paramV = param_nu[:, 0]
    nu = param_nu[:, 1]

    # Make figure
    import lepm.plotting.plotting as leplt
    fig, ax = leplt.initialize_1panel_centered_fig(wsfrac=0.5, tspace=4)

    if xlabel is None:
        xlabel = param.replace('_', ' ')

    # Plot the curve
    # first sort the paramV values
    if isinstance(paramV[0], float):
        si = np.argsort(paramV)
    else:
        si = np.arange(len(paramV), dtype=int)
    ax.plot(paramV[si], nu[si], '.-')
    ax.set_xlabel(xlabel)

    # Add title
    ax.text(0.5, 0.95, title, transform=fig.transFigure, ha='center', va='center')

    # Save the plot
    if param_type == 'glat':
        outdir = rootdir + 'kspace_cherns_gyro/chern_glatparam/' + param + '/'
        dio.ensure_dir(outdir)

        # Add meshfn name to the output filename
        outbase = self.cherns[self.cherns.items()[0][0]][0].gyro_lattice.lp['meshfn']
        if outbase[-1] == '/':
            outbase = outbase[:-1]
        outbase = outbase.split('/')[-1]

        outd_ex = self.cherns[self.cherns.items()[0][0]][0].gyro_lattice.lp['meshfn_exten']
        # If the parameter name is part of the meshfn_exten, replace its value with XXX in
        # the meshfnexten part of outdir.
        mfestr = glatfns.param2meshfnexten_name(param)
        if mfestr in outd_ex:
            'param is in meshfn_exten, splitting...'
            # split the outdir by the param string
            od_split = outd_ex.split(mfestr)
            # split the second part by the value of the param string and the rest
            od2val_rest = od_split[1].split('_')
            odrest = od_split[1].split(od2val_rest[0])[1]
            print 'odrest = ', odrest
            print 'od2val_rest = ', od2val_rest
            outd_ex = od_split[0] + param + 'XXX'
            outd_ex += odrest
            print 'outd_ex = ', outd_ex
        else:
            outd_ex += '_' + param + 'XXX'
    elif param_type == 'lat':
        outdir = rootdir + 'kspace_cherns_gyro/chern_lpparam/' + param + '/'
        outbase = self.cherns[self.cherns.items()[0][0]][0].gyro_lattice.lp['meshfn']
        # Take apart outbase to parse out the parameter that is varying
        mfestr = latfns.param2meshfnexten_name(param)
        if mfestr in outbase :
            'param is in meshfn_exten, splitting...'
            # split the outdir by the param string
            od_split = outbase.split(mfestr)
            # split the second part by the value of the param string and the rest
            od2val_rest = od_split[1].split('_')
            odrest = od_split[1].split(od2val_rest[0])[1]
            print 'odrest = ', odrest
            print 'od2val_rest = ', od2val_rest
            outd_ex = od_split[0] + param + 'XXX'
            outd_ex += odrest
            print 'outd_ex = ', outd_ex
        else:
            outbase += '_' + param + 'XXX'

        outd_ex = self.cherns[self.cherns.items()[0][0]][0].gyro_lattice.lp['meshfn_exten']

    dio.ensure_dir(outdir)

    fname = outdir + outbase
    fname += '_chern_' + param + '_Ncoll' + '{0:03d}'.format(len(self.gyro_collection.gyro_lattices))
    fname += outd_ex
    print 'saving to ' + fname + '.png'
    plt.savefig(fname + '.png')
    plt.clf()