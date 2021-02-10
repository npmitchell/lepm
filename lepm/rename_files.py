# !/usr/bin/python
import os, sys
import glob
import argparse

# from os import getcwd,chdir

'''Replace unique character substring in all filenames or directory names in a specified directory.
If a space is required in either string, use '-space-'.

Working examples
----------------
python rename_files.py -oldstr abc__apostrophe__s__space__abc__closedparen__.png -newstr .png -dir /Users/npmitchell/Dropbox/Soft_Matter/PAPER/nanoparticle/figure_drafting/fig_simulation_render_houdini/render_S082/
python rename_files.py -oldstr 100pps -newstr 4000pps -dir /Volumes/labshared3/noah/turbulence/vortex_scaling_lateral/mask008/mask008_0p3hz/20180719_mask008_0p3hz/piston/

Example usage (all valid):
--------------------------
python rename_files.py -oldstr textfile -newstr nicefile -dir ./directory_to_change
python rename_files.py -oldstr textfile -newstr nicefile -dir ./directory_to_change/
python rename_files.py -oldstr textfile -newstr nicefile -dir /Users/labuser/directory_to_change
python rename_files.py -oldstr textfile -newstr nicefile -dir /Users/labuser/directory_to_change/
python rename_files.py -oldstr textfile -newstr nicefile -dir ./directory_to_change/,./other_dir_to_change/

Will NOT work:
python rename_files.py -oldstr textfile -newstr nicefile -dir ../directory_to_change/
python rename_files.py -oldstr textfile -newstr nicefile -dir ../../directory_to_change/
'''


def rename_files(directory, oldstr, newstr):
    """Replace all the filenames in directory containing oldstr with newstr.
    
    Parameters
    ----------
    directory : string
        The directory in which to look for files with filenames containing oldstr
    oldstr : string
        string to replace in filenames
    newstr : string
        string to put in place of oldstr in filenames
    """
    # Get directory name from input arguments
    pwd = os.getcwd() + '/'
    directory = directory.replace('./', pwd)
    if directory[-1] != '/':
        directory += '/'

    oldstr = oldstr.replace('__space__', ' ')
    oldstr = oldstr.replace('space__', ' ')
    oldstr = oldstr.replace('__openparen__', '(')
    oldstr = oldstr.replace('openparen__', '(')
    oldstr = oldstr.replace('__closedparen__', ')')
    oldstr = oldstr.replace('__apostrophe__', "'")

    newstr = newstr.replace('__space__', ' ')

    # check if special characters like parentases or brackets are already bound by brackets
    if oldstr.split('(')[-1][0] != ']':
        oldstr = oldstr.replace('(', '[(]').replace(')', '[)]')
        newstr = newstr.replace('(', '[(]').replace(')', '[)]')

    print "The directory is: %s" % directory
    print "The files to be replaced are is: %s" % glob.glob(directory + '*' + oldstr + '*')

    # Note below we avoid overwritting THIS file
    # (which could be changed if looking in our present working directory).
    renamed = False
    print 'oldstr = ', oldstr
    print 'glob.glob(directory + * + oldstr + *) = ', glob.glob(directory + "*" + oldstr + "*")
    for filename in glob.glob(directory + "*" + oldstr + "*"):
        if not filename == 'rename_files.py':
            # renaming files in directory
            # Convert special characters back into literal strings
            oldstr = oldstr.replace('[(]', '(').replace('[)]', ')')
            if oldstr in filename:
                newname = filename.split(oldstr)[0] + newstr + filename.split(oldstr)[1]
                os.rename(filename, newname)
                print 'filename = ', filename
                print 'new filename = ', newname
                renamed = True

    if renamed:
        print "Successfully renamed."
    else:
        print "Did not rename any files."

    # listing directories after renaming "tutorialsdir"
    print "the dir now contains: %s" % os.listdir(directory)


if __name__ == '__main__':
    # Parse arguments from command line
    parser = argparse.ArgumentParser(description='Specify time string (timestr) for gyro simulation.')
    parser.add_argument('-dir', '--directory', help='Name of directory inside which to rename files.' +
                                                    'If not absolute path, use ./ to denote pwd. ' +
                                                    'Currently cannot contain ../', type=str, default='./')
    parser.add_argument('-oldstr', '--oldstr', help='String to delete in file names', type=str, default='')
    parser.add_argument('-newstr', '--newstr', help='String to add in file names in place of delete',
                        type=str, default='')
    args = parser.parse_args()

    # For each directory specified by dir, run rename_files
    dirs = args.directory.split(',')

    for directory in dirs:
        rename_files(directory, args.oldstr, args.newstr)
