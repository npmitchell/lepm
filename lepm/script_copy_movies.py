import glob
import lepm.lattice_elasticity as le
import subprocess

lt = 'kagper_hucent'
rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/chern/'
rootdir += lt
dest = '/Users/npmitchell/Dropbox/Soft_Matter/PAPER/gyro_disorder_paper/videos/'
# rootdir = '/Volumes/research2TB/test/'
subdirs = le.find_subdirs(lt + '*40', rootdir)
print 'subdirs = ', subdirs

for sd in subdirs:
    print 'sd = ', sd
    name = sd.split('/')[-2]
    moviefn = glob.glob(sd + '*Nks201*.mov')
    if moviefn:
        movname = moviefn[0].split('/')[-1]
        print 'dest = ', dest
        print 'name = ', name
        print 'movname = ', movname
        destfn = dest + name + '_' + movname
        print 'moviefn = ', moviefn[0]
        print 'calling: ', 'cp', moviefn[0], destfn
        subprocess.call(['cp', moviefn[0], destfn])
