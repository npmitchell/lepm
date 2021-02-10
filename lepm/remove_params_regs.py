import glob
import lepm.lattice_elasticity as le
import subprocess

rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/chern/'
rootdir += 'twisted_kagome/'
# rootdir = '/Volumes/research2TB/test/'
subdir, subsubdir = le.find_subsubdirectory('Omkn1p00_Omgn1p00_*', rootdir)
print 'subsubdir = ', subsubdir

for ssd in subsubdir:
    trash, prdir = le.find_subsubdirectory('params_regs', ssd)
    print 'prdir = ', prdir
    if prdir:
        for p2del in prdir:
            print 'calling: ', 'rm', '-r', p2del
            subprocess.call(['rm', '-r', p2del])

    prdir = le.find_subdirs('params_regs', ssd)
    contribdir = le.find_subdirs('contribs', ssd)
    print 'prdir = ', prdir
    if prdir and not contribdir:
        for p2del in prdir:
            print 'calling: ', 'rm', '-r', p2del
            subprocess.call(['rm', '-r', p2del])

    prdir = le.find_subdirs('visualization', ssd)
    print 'prdir = ', prdir
    if prdir and not contribdir:
        for p2del in prdir:
            print 'calling: ', 'rm', '-r', p2del
            subprocess.call(['rm', '-r', p2del])

