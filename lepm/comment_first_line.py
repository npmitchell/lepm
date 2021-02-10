import glob
import argparse

"""
Open all files in a directory. Place a '#' in front of the first line if it contains more than N elements
(delimited by spaces).

Example:
python comment_first_line.py -seriesdir /Users/npmitchell/Dropbox/Soft_Matter/GPU/networks/jammed_source_hexner_polydisperse/N000064 -fn XY*.txt -thresN 4
python comment_first_line.py -seriesdir /Users/npmitchell/Dropbox/Soft_Matter/GPU/networks/hyperuniform_source/hyperuniform_N030 -fn out*_xy.txt -thresN 4
"""

parser = argparse.ArgumentParser(description='Specify parameters for commenting out first line in text files, in batch.')
parser.add_argument('-lineN', '--line_number', help='The line to comment, usually zero (0)', type=int, default=0)
parser.add_argument('-thresN', '--thresN_elems',
                    help='The minimum # elements for the line to have to mandate commenting it',
                    type=int, default=5)
parser.add_argument('-seriesdir', '--seriesdir',
                    help='Directory in which to search for text files (or subdirs with text files)',
                    type=str, default='./')
parser.add_argument('-subdir', '--subdir_name', help='Subdirectory name in which to search for text files',
                    type=str, default='series*')
parser.add_argument('-fn', '--filename', help='File name to search for, with wildcard in it',
                    type=str, default='file*.txt')
parser.add_argument('-do_subdirs', '--do_subdirs',
                    help='If true, goes through all subdirs of seriesdir. Otherwise looks for files in seriesdir.',
                    action='store_true')
args = parser.parse_args()


def comment_header(file_name, line_num, thresN, delimiter=' '):
    lines = open(file_name, 'r').readlines()
    if len(lines[line_num].split(delimiter)) > thresN - 1:
        if lines[line_num][0] != '#':
            print 'commenting line ', line_num, ' in ', file_name, '...'
            lines[line_num] = '# ' + lines[0]
            out = open(file_name, 'w')
            out.writelines(lines)
            out.close()

if args.do_subdirs:
    # List subdirs, look for text files in subdirs
    seriesdir = args.seriesdir
    if seriesdir[-1] != '/':
        seriesdir += '/'

    print 'Looking in seriesdir = ', seriesdir
    subdirs = glob.glob(seriesdir + args.subdir_name)
    print 'found subdirs = ', subdirs

    for subdir in subdirs:
        files = glob.glob(subdir + '/' + args.filename)
        print 'found files = ', files
        for ff in files:
            comment_header(ff, args.line_number, args.thresN_elems)

else:
    # Look for text files in seriesdir
    seriesdir = args.seriesdir
    if seriesdir[-1] != '/':
        seriesdir += '/'

    print 'Looking in seriesdir = ', seriesdir
    files = glob.glob(seriesdir + args.filename)
    print 'found files = ', files
    for ff in files:
        comment_header(ff, args.line_number, args.thresN_elems)
