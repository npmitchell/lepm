import glob
import lepm.plotting.movies as lemov
import lepm.plotting.plotting as leplt
import subprocess

dir = '/Users/npmitchell/Desktop/for_movie/'
new_basename = 'image'
# movframedir = dir[:-1] + '_mov/'
ims = sorted(glob.glob(dir + '*.png'))

ii = 0
for im in ims:
    basename = im.split('.png')[0].split('/')[-1][:-5]
    rootname = im.split(basename)[0]
    if basename != new_basename:
        newname = rootname + new_basename + '{0:05d}'.format(ii) + '.png'
        print 'moving ', im, ' to ', newname
        subprocess.call(['mv', im, newname])

    # Place image on a canvas that is the correct size for making an ffmpeg movie
    # fig, ax, header_ax, header_cbar, ax_cbar = \
    #     leplt.initialize_landscape_with_header(preset_cbar=False, ax_pos=[0.0, 0.0, 1., 1.])
    # imf = plt.
    # ax.imshow()

    ii += 1

imgname = rootname + new_basename
movname = dir + 'twist_spring.mov'
lemov.make_movie(imgname, movname, framerate=9)