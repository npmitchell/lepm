import subprocess
import argparse
import numpy as np

'''
Run a series of simulations, each with a one parameter different between them.
Example usage:

# make lattice
python run_series.py -pro make_lattice -opts NV/19/-theta/0.5 -var NH 33/35/37/39/41/43/45/47/49
python run_series.py -pro make_lattice -opts N/6/-eta/0.6/-shape/hexagon -var delta 0.667/0.750/0.850/0.950/1.050/1.150/1.250/1.350/0.995
python run_series.py -pro make_lattice -opts N/15/-eta/0.0/-shape/hexagon -var delta 0.667/0.750/0.850/0.950/1.050/1.150/1.250/1.350/0.995
python run_series.py -pro make_lattice -opts N/15/-eta/0.0/-shape/hexagon -var delta 0.700/0.800/0.900/1.100/1.200/1.300/1.350/0.995
python run_series.py -pro make_lattice -opts NV/9/-theta/0.5 -var NH 3/5/7/9/11/13/15/17/19/21/23/25/27/29/31/33/35/37/39/41/43/45/47/49/51/53/55/57/59/61/63/65/67/69/71
python run_series.py -pro make_lattice -opts N/10/-eta/1.0/-shape/hexagon/-nice_plot/-skip_gyroDOS/-skip_massDOS -var delta 0.667/0.700/0.750/0.800/0.850/0.900/0.950/1.050/1.100/1.150/1.200/1.250/1.300/1.350/0.995
python run_series.py -pro make_lattice -opts N/20/-shape/hexagon -var delta 0.700/0.750/0.800/0.850/0.900/0.950/1.100/1.050/1.100/1.150/1.200/1.250/1.300/1.350/0.995
python run_series.py -pro make_lattice -opts N/30/-LT/kagper_hucent/-skip_massDOS -var perd 0.0:0.1:1.0
python run_series.py -pro make_lattice -opts N/50/-LT/kagome_hucent -var conf 02:8
python run_series.py -pro make_lattice -opts N/15/-LT/kagper_hex -var -perd 0.25:0.25:1.0

# make twisted kagome lattices
python run_series.py -pro make_lattice -opts N/10/-LT/twisted_kagome/-modii/1000/-TS/3000 -var alph 0.1:0.1:1.1

# Make percolation sample -- single conf
python run_series.py -pro make_lattice -opts N/30/-LT/kagper_hucent/-DOSmovie/-skip_massDOS -var perd 0.01:0.01:0.11
python run_series.py -pro make_lattice -opts N/30/-LT/kagper_hucent/-DOSmovie/-skip_massDOS/-conf/02 -var perd 0.1:.1:1.0

# Make percolation sample -- many confs, single perd
python run_series.py -pro make_lattice -opts N/40/-LT/kagper_hucent/-perd/0.75/-skip_bondL_hist -var conf 3:11

# Make Periodic BC hyperuniform samples
python run_series.py -pro make_lattice -opts NP/20/-LT/hucentroid/-periodic/-DOSmovie -var conf 01:10
python run_series.py -pro make_lattice -opts NP/50/-LT/hucentroid/-periodic/-DOSmovie -var conf 01:30
python run_series.py -pro make_lattice -opts NP/20/-LT/hucentroid/-periodic/-DOSmovie -var conf 01:10


# kitaev chern calc with disorder
python run_series.py -pro kitaev_chern_gyro_calc_finitesize_effect -opts LT/hexagonal/-skip_DOS_ims/-eta/0.6/-N/6/-shape/hexagon/-ksize/0.1:0.2:1.3 -var delta 0.667/0.750/0.850/0.950/1.050/1.150/1.250/1.350/0.995

# kitaev chern calc near edge -- localization effects
python run_series.py -pro kitaev_chern_gyro_calc_finitesize_effect -opts N/20/-LT/hexagonal/-shape/square/-ksize/0.3:0.05:0.6/-delta/0.750 -var poly_offset 05.0_00.0/10.0_00.0/12.0_00.0/12.5_00.0/13.0_00.0/13.5_00.0/14.0_00.0/14.5_00.0/15.0_00.0/15.5_00.0/16.0_00.0/16.5_00.0/17.0_00.0/17.5_00.0/18.0_00.0/18.5_00.0/19.0_00.0/19.5_00.0
python run_series.py -pro kitaev_chern_gyro_calc_finitesize_effect -opts N/20/-LT/hexagonal/-shape/square/-ksize/0.3:0.05:0.6/-delta/0.750/-polyT -var poly_offset 00.0_05.0/00.0_10.0/00.0_10.5/00.0_11.0/00.0_11.5/00.0_12.0/00.0_12.5/00.0_13.0/00.0_13.5/00.0_14.0/00.0_14.5/00.0_15.0/00.0_15.5
python run_series.py -pro kitaev_chern_gyro_calc_finitesize_effect -opts N/20/-LT/hexagonal/-shape/square/-ksize/0.3:0.05:0.6/-delta/0.850 -var poly_offset 05.0_00.0/14.0_00.0/14.5_00.0/15.0_00.0/15.5_00.0/16.0_00.0/16.5_00.0/17.0_00.0/17.5_00.0/18.0_00.0/18.5_00.0/19.0_00.0/19.5_00.0/20.0_00.0
python run_series.py -pro kitaev_chern_gyro_calc_finitesize_effect -opts N/20/-LT/hexagonal/-shape/square/-ksize/0.3:0.05:0.6/-delta/0.850 -var poly_offset 00.0_07.0/00.0_08.0/00.0_09.0/   00.0_00.0/00.0_05.0/00.0_10.0/00.0_10.5/00.0_11.0/00.0_11.5/00.0_12.0/00.0_12.5/00.0_13.0/00.0_13.5/00.0_14.0
python run_series.py -pro kitaev_chern_gyro_calc_finitesize_effect -opts N/20/-LT/hexagonal/-shape/square/-ksize/0.3:0.05:0.6/-delta/0.850/-polyT -var poly_offset 00.0_00.0/00.0_05.0/00.0_07.0/00.0_08.0/00.0_09.0/00.0_10.0/00.0_10.5/00.0_11.0/00.0_11.5/00.0_12.0/00.0_12.5/00.0_13.0

python run_series.py -pro kitaev_chern_gyro_calc_finitesize_effect -opts N/20/-LT/hexagonal/-shape/square/-ksize/0.3:0.05:0.6/-delta/0.900 -var poly_offset 00.0_00.0/05.0_00.0/14.0_00.0/14.5_00.0/15.0_00.0/15.5_00.0/16.0_00.0/16.5_00.0/17.0_00.0/17.5_00.0/18.0_00.0/18.5_00.0/19.0_00.0/19.5_00.0/20.0_00.0
python run_series.py -pro kitaev_chern_gyro_calc_finitesize_effect -opts N/20/-LT/hexagonal/-shape/square/-ksize/0.3:0.05:0.6/-delta/0.900 -var poly_offset 00.0_00.0/00.0_05.0/00.0_07.0/00.0_08.0/00.0_09.0/00.0_10.0/00.0_10.5/00.0_11.0/00.0_11.5/00.0_12.0/00.0_12.5/00.0_13.0/00.0_13.5/00.0_14.0
python run_series.py -pro kitaev_chern_gyro_calc_finitesize_effect -opts N/20/-LT/hexagonal/-shape/square/-ksize/0.3:0.05:0.6/-delta/0.900/-polyT -var poly_offset 00.0_00.0/00.0_05.0/00.0_07.0/00.0_08.0/00.0_09.0/00.0_10.0/00.0_10.5/00.0_11.0/00.0_11.5/00.0_12.0/00.0_12.5/00.0_13.0

python run_series.py -pro kitaev_chern_gyro_calc_finitesize_effect -opts N/10/-LT/hexagonal/-shape/square/-ksize/0.6:0.1:1.2 -var poly_offset 00.0_00.0/05.0_00.0/05.5_00.0/06.5_00.0/07.0_00.0/07.5_00.0/08.0_00.0/08.5_00.0/09.0_00.0
python run_series.py -pro kitaev_chern_gyro_calc_finitesize_effect -opts N/10/-LT/hexagonal/-shape/square/-ksize/0.6:0.1:1.2/-polyT -var poly_offset 00.0_00.0/00.0_04.0/00.0_04.5/00.0_05.0/00.0_05.5/00.0_06.0/00.0_06.5/00.0_07.0/00.0_07.5/00.0_08.0
python run_series.py -pro kitaev_chern_gyro_calc_finitesize_effect -opts N/10/-LT/hexagonal/-shape/square/-ksize/0.6:0.1:1.2/-delta/0.750 -var poly_offset 00.0_00.0/04.0_00.0/04.5_00.0/05.0_00.0/05.5_00.0/06.0_00.0/06.5_00.0/07.0_00.0/07.5_00.0/08.0_00.0/08.5_00.0/09.0_00.0/09.5_00.0/10.0_00.0/10.5_00.0/11.0_00.0/11.5_00.0
python run_series.py -pro kitaev_chern_gyro_calc_finitesize_effect -opts N/10/-LT/hexagonal/-shape/square/-ksize/0.6:0.1:1.2/-delta/0.750/-polyT -var poly_offset 00.0_00.0/00.0_04.0/00.0_04.5/00.0_05.0/00.0_05.5/00.0_06.0/00.0_06.5/00.0_07.0/00.0_07.5/00.0_08.0
python run_series.py -pro kitaev_chern_gyro_calc_finitesize_effect -opts N/10/-LT/hexagonal/-shape/square/-ksize/0.6:0.1:1.2/-delta/0.850 -var poly_offset 00.0_00.0/04.0_00.0/04.5_00.0/05.0_00.0/05.5_00.0/06.0_00.0/06.5_00.0/07.0_00.0/07.5_00.0/08.0_00.0/08.5_00.0/09.0_00.0/09.5_00.0/10.0_00.0/10.5_00.0/11.0_00.0/11.5_00.0
python run_series.py -pro kitaev_chern_gyro_calc_finitesize_effect -opts N/10/-LT/hexagonal/-shape/square/-ksize/0.6:0.1:1.2/-delta/0.850 -var poly_offset 00.0_00.0/00.0_04.0/00.0_04.5/00.0_05.0/00.0_05.5/00.0_06.0/00.0_06.5/00.0_07.0/00.0_07.5/00.0_08.0
python run_series.py -pro kitaev_chern_gyro_calc_finitesize_effect -opts N/10/-LT/hexagonal/-shape/square/-ksize/0.6:0.1:1.2/-delta/0.850/-polyT -var poly_offset 00.0_00.0/00.0_04.0/00.0_04.5/00.0_05.0/00.0_05.5/00.0_06.0/00.0_06.5/00.0_07.0/00.0_07.5/00.0_08.0


# kitaev chern calc through hexagonal top transition
python run_series.py -pro kitaev_chern_gyro_calc_finitesize_effect -opts LT/hexagonal/-shape/hexagon/-N/10/-skip_DOS_ims/-ksize/0.0:0.1:1.20 -var delta 0.666/0.700/0.750/0.800/0.850/0.900/0.950/1.050/1.100/1.150/1.200/1.250/1.300/1.350/0.995
python run_series.py -pro kitaev_chern_gyro_calc_finitesize_effect -opts LT/hexagonal/-shape/hexagon/-N/10/-eta/1.0/-skip_DOS_ims/-ksize/0.0:0.1:1.20 -var delta 0.667/0.700/0.750/0.800/0.850/0.900/0.950/1.050/1.100/1.150/1.200/1.250/1.300/1.350/0.995
python run_series.py -pro kitaev_chern_gyro_calc_finitesize_effect -opts LT/hexagonal/-shape/hexagon/-N/15/-eta/0.0/-skip_DOS_ims/-ksize/0.0:0.1:1.20 -var delta 0.667/0.700/0.750/0.800/0.850/0.900/0.950/1.050/1.100/1.150/1.200/1.250/1.300/1.350/0.995
python run_series.py -pro kitaev_chern_gyro_calc_finitesize_effect -opts LT/hexagonal/-shape/hexagon/-N/20/-eta/0.0/-skip_DOS_ims/-ksize/0.0:0.1:1.20 -var delta 0.667/0.700/0.750/0.800/0.850/0.900/0.950/1.050/1.100/1.150/1.200/1.250/1.300/1.350/0.995

# New way of kitaev_chern calc through hexagonal -> bowtie topological transition
python run_series.py -pro kitaev_collection -opts LT/hexagonal/-N/15/-omegac/2.25/- -var

# kitaev chern calc through omegac vals
python run_series.py -pro kitaev_chern_gyro_calc_finitesize_effect -opts LT/penroserhombTricent/-shape/circle/-N/09/-skip_DOS_ims/-ksize/0.0:0.1:1.80 -var omegac 2.171/2.351/2.600
python run_series.py -pro kitaev_chern_gyro_calc_finitesize_effect -opts LT/penroserhombTricent/-shape/circle/-N/20/-skip_DOS_ims/-ksize/0.1:0.1:1.20 -var omegac 2.171/2.351/2.600
python run_series.py -pro kitaev_chern_gyro_calc_finitesize_effect -opts LT/penroserhombTricent/-shape/circle/-N/30/-skip_DOS_ims/-ksize/0.1:0.1:1.20 -var omegac 2.171/2.351/2.600
2.170682198272 / 2.350787658607 / 2.599677160914
python run_series.py -pro kitaev_collection -opts LT/hucentroid/-shape/square/-N/50/-vary_omegac/-omegac/1.0:.1:4.0 -var conf 02:10

# kitaev chern percolation
python run_series.py -pro kitaev_chern_gyro_calc_finitesize_effect -opts LT/kagper_hucent/-N/30/-shape/square/-omegac/2.250/-sqrt_ksizestep/-ksize/0.0:0.01:1.2 -var perd 0.0:0.1:1.1
python run_series.py -pro kitaev_collection -opts LT/kagper_hucent/-N/20/-vary_omegac/-omegac/1.0:.1:4.0 -var perd 0.0:0.1:1.1
python run_series.py -pro kitaev_collection -opts LT/kagper_hucent/-N/30/-vary_omegac/-omegac/1.0:.1:4.0/-conf/02 -var perd 0.0:0.1:1.1

# varyloc series
python run_series.py -pro kitaev_collection -opts LT/kagper_hucent/-N/40/-varyloc/-Nks/201/-perd/0.25 -var conf 01:01:11
running on midway:
python run_series.py -pro kitaev_collection -opts LT/kagper_hucent/-N/40/-varyloc/-Nks/201/-perd/0.25 -var conf 01:02:11
python run_series.py -pro kitaev_collection -opts LT/kagper_hucent/-N/40/-varyloc/-Nks/201/-perd/0.25 -var conf 03:02:11
python run_series.py -pro kitaev_collection -opts LT/kagper_hex/-N/15/-varyloc/-Nks/201 -var perd 0.25:0.25:1.0

# DOS movies for series
python run_series.py -pro gyro_lattice_class -opts DOSmovie/-LT/kagper_hucent/-N/40/-perd/0.5 -var conf 1:1:10

# Chern disorder and deformation angle phase portrait
python run_series.py -pro
kitaev_chern_class.py -LT hexagonal -N 6 -Vpin 0.3
'''

parser = argparse.ArgumentParser('Run a series of programs, one by one, with different arguments')
parser.add_argument('-pro', '--program', help='Name of program to run many times with different options',
                    type=str, default='gyro_lattice_1stO_torque')
parser.add_argument('-opts', '--static_options', help='String to run for every sim, with / chars instead of spaces',
                    type=str, default='')
parser.add_argument('-var', '--vary_var', help='Variable to change with each sim', type=str, default='check_empty')
parser.add_argument('vals', type=str, nargs='?',
                    help='values to use as var for each simulation, as val0/val1/val2/etc, OR as int:int:int or' +
                         ' float:float:float or int:int or float:float',
                    default='check_string_for_empty')
    
args = parser.parse_args()

if args.vals == 'check_string_for_empty':
    raise RuntimeError('ERROR! No values to assign.')
elif ':' in args.vals:
    values = args.vals.split(':')
    if '.' in values[0] or '.' in values[1]:
        print 'Supplied values are floats...'
        print 'values = ', values
        start = float(values[0].replace('n', '-'))
        if len(values) == 3:
            step = float(values[1].replace('n', '-'))
            end = float(values[2].replace('n', '-'))
        elif len(values) == 2:
            step = float(1)
            end = float(values[1].replace('n', '-'))
        else:
            raise RuntimeError('If : is used, vals must be ##:## or ##:##:##')
    else:
        print 'Values are integers...'
        start = int(values[0].replace('n', '-'))
        if len(values) == 3:
            step = int(values[1].replace('n', '-'))
            end = int(values[2].replace('n', '-'))
        elif len(values) == 2:
            print 'Incrementing by 1 between each subprocess call...'
            step = 1
            end = int(values[1].replace('n', '-'))
        else:
            raise RuntimeError('If : is used, vals must be ##:## or ##:##:##')
    print 'setting vals_nums to np.arange(', start, ',', end, ',', step, ')...'
    vals_nums = np.arange(start, end, step)
    vals = [str(vali) for vali in vals_nums]
else:
    vals = np.array([args.vals.split('/')[i] for i in range(len(args.vals.split('/')))])
    
# print vals

if args.static_options == '':
    optlist = []
else:
    optlist = args.static_options.split('/')
    optlist[0] = '-'+optlist[0]

print "optlist = ", optlist

# An example string fed into subprocess is:
# python gyro_lattice_1stO_torque.py -NH 13 -NV 19 -theta 0.5 -split_\
#       spin -split_k 0.15 -freq 2.290181636810 -TS 30000 -modii 200 -excite_dist 4.0

for val in vals:
    if args.vary_var == 'check_empty':
        print 'val= ', val
        callstr = ['python', args.program+'.py'] + optlist + [val]
    else:
        callstr = ['python', args.program+'.py'] + optlist + ['-' + args.vary_var, val]
    print 'Calling subprocess: \n', callstr
    subprocess.call(callstr)
