import numpy as np
import matplotlib.pyplot as plt



def change_axes_geometry_stack(fig, ax, naxes):
    """Take a figure with stacked axes and add more stacked (in Y) axes to the figure, shuffling the others
    to make room for the new one(s).
    """
    for ii in range(len(ax)):
        geometry = (naxes,1,ii+1)
        if ax[ii].get_geometry() != geometry:
            ax[ii].change_geometry(*geometry)
        plt.pause(1)

    for ii in np.arange(len(ax),naxes):
        print 'adding axis ', ii
        fig.add_subplot(naxes, 1, ii+1)
        plt.pause(1)

    ax = fig.axes
    return fig, ax


fig, ax = plt.subplots(6, 1) #, sharex=True, sharey=True)
ax[0].plot([1,2,3,2.4],[5,3,2,1],'b.-')
ax[1].plot([1,2,3,2.4],[5,3,2,1],'r.-')
ax[2].plot([1,2,3,2.4],[5,3,2,1],'g.-')
ax[3].plot([1,2,3,2.4],[5,3,2,1],'c.-')
plt.show()


fig, ax = plt.subplots(6, 1)
ax[0].plot([1,2,3,2.4],[5,3,2,1],'b.-')
ax[1].plot([1,2,3,2.4],[5,3,2,1],'r.-')
ax[2].plot([1,2,3,2.4],[5,3,2,1],'g.-')
ax[3].plot([1,2,3,2.4],[5,3,2,1],'c.-')
fig, ax = change_axes_geometry_stack(fig, ax, 8)
plt.show()

# new_fig = plt.figure()
# geometry = (7,1,4)
# ax[3].change_geometry(*geometry)
# new_fig.axes.append(ax[3])



