import numpy as np
from matplotlib.path import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.integrate import dblquad
import cPickle as pickle
from matplotlib import cm
import time
import itertools
import os
import os.path
from scipy.interpolate import griddata
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage
import Chern_calc.chern_functions_gen as cf
import Chern_calc.BZ_funcs as BZ
import lattice_functions as lf
import science_plot_style as sps
import Chern_calc.view_plot as vp
import Chern_calc.test_lattice as tl
import Chern_calc.BZ_funcs as bz
import sys
import lattice_elasticity as le
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
#from make_lattice import *
import scipy.spatial as spatial
import glob
from scipy.spatial import Delaunay

def latticevec_filter(BL,xy, C, CBL, inv=False):
    '''Filter out bonds from BL based on whether the bonds lie along vecs defined by bonds in CBL (unit cell bond list)
    
    Parameters
    ----------
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points
    xy : array of dimension nx2
        2D lattice of points (positions x,y)
    C : array of dimension #particles in unit cell x 2
        unit cell positions (xy)
    CBL : array of dim #bonds in unit cell x 2
        bond list for unit cell
                
    Returns
    ----------
    BLout : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points (extraneous bonds removed)
    '''
    print 'inv is', inv
    print 'len is', len(xy)
    Cvecs_pos = np.array([ C[CBL[i,1],:] - C[CBL[i,0],:] for i in range(len(CBL))])
    Cvecs = np.vstack((Cvecs_pos,-Cvecs_pos))

    #if bond in BL does not match Cvec, then cut it
    vecs = np.array([ xy[BL[i,1],:] - xy[BL[i,0],:] for i in range(len(BL))])
    

    vecmatch = np.zeros(len(BL),dtype=bool)
    for i in range(len(BL)):

        vecmatch[i] = any( (abs(Cvecs[:]-vecs[i])<1e-1).all(1) )
    
    if inv:
        vecmatch = np.invert(vecmatch)
    
    #vecmatch = any(vecmatchM,0)
    BLout = BL[vecmatch]

    return BLout

def make_vis(xy, a, BL, sbl):
    R = xy
    a1 = np.array([a[2][0], a[2][1], 0])
    a2 = np.array([a[1][0], a[1][1], 0])
    vtx, vty = bz.find_BZ_zone(a1, a2)
    
    vertex_points = np.array([vtx, vty]).T
    
    c1 = '#D8AC51'
    c2 = '#3B53E9'
    c3 = '#A72E3C'
    c4 = '#39A7A7'
    
    cc =[c1, c2, c3, c4]
    
    Ni, Nk = le.BL2NLandKL(BL,nn=10)
    
    line_cols = ['k' for i in range(len(BL))]
    
    cols = []
    for i in range(len(sbl)):
        cols.append(cc[sbl[i]])
    
    
    return np.array(R), np.array(Ni), np.array(Nk), cols, line_cols, vertex_points

def automatic_semiinfinite_cell(NH, NV, C, CBL, sbl, a, check = False, get_BL = False):

    CU = np.arange(len(C), dtype='int')
    a = a[1:]
    a1 = a[0]
    a2 = a[0]
    #translate by Bravais latt vecs
    print('Translating by Bravais lattice vectors...')
    LV  = [a1, a2]
    inds = np.arange(len(C))
    tmp0 = np.zeros(len(C), dtype='int')
    tmp1 = np.ones(len(C), dtype='int')
    
    for i in np.arange(NV):
        for j in np.arange(NH):
            # bottom row
            if i == 0:
                if j == 0 :
                    Bprev = C
                    R = C
                    SBL = sbl.tolist()
                    LVUC = np.dstack((tmp0,tmp0,CU ))[0]
                    
                
                else:
                    R = np.vstack((R, C + j*LV[0]))
                    SBL += sbl.tolist() #np.vstack((SBL, sbl))

                    LVUCadd = np.dstack((j*tmp1,tmp0,CU ))[0]
                    #print 'LVUCadd = ', LVUCadd
                    LVUC = np.vstack((LVUC,LVUCadd))
                    
                 
            else:
                # vertical indices to copy over
                vinds = np.array([1,2,5,6,7,8])
                R = np.vstack((R, C[:] + j*LV[0] + i*LV[1]))
                SBL += sbl.tolist()
                LVUCadd = np.dstack((j*tmp1[:],i*tmp1[:],CU[:] ))[0]
                #print 'LVUCadd = ', LVUCadd
                LVUC = np.vstack((LVUC,LVUCadd))
                    
              
    
    SBL = np.asarray(SBL)
     #get rid of repeated points
 
    print('Eliminating repeated points...')
    #########
    # check
    if check:
        plt.clf()
        sizevals = np.arange(len(R))+10
        colorvals = np.linspace(0,1,len(R))
        plt.scatter(R[:,0],R[:,1],s=sizevals,c=colorvals, cmap='afmhot', alpha=0.2)
        plt.show()
    #########

    #R, si, ui = le.args_unique_rows_threshold(R, 0.2)
    R, si, ui = le.eliminate_R_based_on_distance(R, 0.05)

    #########
    # check
    if check:
        sizevals = np.arange(len(R))+50
        colorvals = np.linspace(0,1,len(R))
        plt.scatter(R[:,0],R[:,1],s=sizevals,c=colorvals, cmap='afmhot', alpha=0.2)
        plt.show()
    #########
    
    SBL = SBL.flatten()
    SBL = SBL[si]
    SBL = SBL[ui]
    xy = R

   
    #xy = R[ui]
    #Triangulate
    print('Triangulating...')
    Dtri = Delaunay(xy)
    btri = Dtri.vertices
   
    #translate btri --> bond list
    BL = le.Tri2BL(btri)
    
    #remove bonds on the sides and through the hexagons
    print('Removing extraneous bonds from triangulation...')
    #calc vecs from C bonds
        
    BL = np.array(BL)
    BL = le.unique_rows(BL) 
    
 
    BL = latticevec_filter(BL, xy, C, CBL)
    
    NL,KL = le.BL2NLandKL(BL,nn=20)    
    
    UC = C
    
    xy[:,0] = xy[:,0] - np.mean(xy[:,0])
    xy[:,1] = xy[:,1] - np.mean(xy[:,1])
    return xy, BL, SBL 

def delete_point(xy, BL, sbl, ind):
    p_array = np.ones(len(xy), dtype = bool)
    p_array[ind] = False

    xy = xy[p_array]
    sbl = sbl[p_array]

    bond_p = np.ones(len(BL), dtype = bool)
    for i in range(len(BL)):
        bond = BL[i]
     
        if bond[0] == ind or bond[1] == ind:

            bond_p[i] = False
    BL = np.array(BL)
    BL = BL[bond_p]

    BL = np.array(BL)
    inds = np.where(BL>ind)
    BL[inds] = BL[inds]-1
    
    return xy, BL, sbl

def remove_ext(xy, BL):
    BList = list(set(np.array(BL).flatten()))
    
    return [i for i in range(len(xy)) if i not in BList]

def calc_semi_in(R, xy, BL, sbl, ons, x_or_y = 'x', hp = 0):
    mid_x = np.mean(xy[:,0])
    mid_y = np.mean(xy[:,1])
    
    point = np.array([mid_x, mid_y]).T

    xy  = xy - point #center xy 
    R = R-point #center R (should be the same center as above.)

    extens = max([max(R[:,0]), min(R[:,0])])
    in_cell = abs(xy[:,0]) < extens 
    
    dir_sbl = list(set(sbl)) #different sublattices
    
    sb_sbl  = []
    rind = []

    #here we will count the number of neighbors for each site.
    bl_fl = list(BL.flatten())
    nn = []
    for j in range(len(xy)):
        nn.append(bl_fl.count(j))
    
    #find how many neighbors a bulk gyro has.
    full_nei = []
    for i in range(len(dir_sbl)):
        sb_sbl.append(xy[sbl ==i ]) #get the members in a certain sublattice
        i_mid = ind_close_to_point(xy, sb_sbl[i], point)
        rind.append(i_mid)
        full_nei.append(nn[i_mid])
    #have the highest number of neighbors.
    
    #find the index in extens with the highest y value for hp sublattice.
    d_ne = full_nei[hp]
    sbl_ind = sbl == hp
    arr = xy[sbl_ind]

    ww =  np.where(arr[:,1] == max(arr[:,1]))[0][0]   
    mxy = arr[ww] #1 if x 0 if y
    dist, start_ind = spatial.cKDTree(xy).query([mxy[0], mxy[1]-5])
    
    #now I have the top starting index rind let's delete evertying above that and below that absolute value of y.
    
    to_del = [i for i in range(len(xy)) if abs(xy[i,1])>= 1.01*abs(xy[start_ind,1])]#abs(xy[:,1])<= 1.01*abs(xy[start_ind,1])
    yval = abs(xy[start_ind,1])
    while len(to_del)>=1:
    
        xy, BL, sbl = delete_point(xy, BL, sbl, to_del[0] )
        to_del = [i for i in range(len(xy)) if abs(xy[i,1])> 1.01*yval]
        
    lw =  np.where(arr[:,1] == min(arr[:,1]))[0][0]   
    mxy = arr[lw] #1 if x 0 if y
    dist, start_ind = spatial.cKDTree(xy).query([mxy[0], mxy[1]+5])
    
    #now let's order all the gyros in each strip by y value
    #we already have in_cell
    #in left cell
    in_cell = np.array([i for i in range(len(xy)) if abs(xy[i,0]) < extens ], dtype = int)
    in_left_cell = np.array([i for i in range(len(xy)) if xy[i,0] < -extens  and xy[i,0] >-3*extens], dtype = int)
    in_right_cell = np.array([i for i in range(len(xy)) if xy[i,0] > extens  and xy[i,0] <3*extens], dtype = int)
  
    xy_in = xy[in_cell]
    BL_in = BL[in_cell]
    sbl_in = sbl[in_cell]
    si = np.argsort(xy_in[:,1])[::-1]
    in_cell = in_cell[si]
    sbl_in = sbl_in[si]
    xy_in = xy_in[si]
    BL_in = BL_in[si]
    
    xy_left = xy[in_left_cell]
    BL_left = BL[in_left_cell]
    sbl_left = sbl[in_left_cell]
    si = np.argsort(xy_left[:,1])[::-1]
    in_left_cell = in_left_cell[si]
    xy_left = xy_left[si]
    BL_left = BL_left[si]

    xy_right = xy[in_right_cell]
    BL_right = BL[in_right_cell]
    sbl_right = sbl[in_right_cell]
    si = np.argsort(xy_right[:,1])[::-1]
    in_right_cell = in_right_cell[si]
    xy_right = xy_right[si]
    BL_right = BL_right[si]

    #okay now they're all in order. We will go through everything in the cell.
    ny = len(in_cell)
    num_neighbors = np.zeros((ny, ny), dtype = int)
    Ni, Nk = le.BL2NLandKL(BL)
    angs = []
    bls = []
    ons_c = []
    tvals = []
    for i in range(ny):
        #find neighbor positions in unit cells.  Don't want to assume anything about y values being different.
        neighs = Ni[in_cell][i][Nk[in_cell][i]>=1]
        ons_c.append(ons[sbl_in[i]])
        mytree = spatial.cKDTree(xy_in)
        dist, index = mytree.query(xy_in[i])
        
        ang_row = []
        bl_row = []
        tv_row =[]
        xy_row = []
        for k in range(len(neighs)):
            #find what number in a long unit cell this neighbor is
            nei_xy = xy[neighs[k]]
            if neighs[k] in list(in_cell):
                mytree = spatial.cKDTree(xy_in)
                dist, indexes = mytree.query(nei_xy)
                xy1 = xy_in[indexes]
              
                
            elif neighs[k] in list(in_left_cell):
                mytree = spatial.cKDTree(xy_left)
                dist, indexes = mytree.query(nei_xy)
                xy1 = xy_left[indexes]
            else: #else must be in right cell
                mytree = spatial.cKDTree(xy_right)
                dist, indexes = mytree.query(nei_xy)
                xy1 = xy_right[indexes]
            
            num_neighbors[index, indexes] += 1
        
            xy0 = xy_in[index]
            vec = xy1-xy0
            ang_row.append(np.arctan2(vec[1], vec[0]))
            bl_row.append(np.linalg.norm(vec))
            tv_row.append(1)
            xy_row.append(xy1)
        xy_row = np.array(xy_row)
        si = np.argsort(xy_row[:,1])[::-1]
        angs.append(np.array(ang_row)[si])
        bls.append(np.array(bl_row)[si])

        tvals.append(list(np.array(tv_row)[si]))

    mM = cf.calc_matrix(angs, num_neighbors, bls= bls, tvals=  list(tvals), ons= ons_c)
    
    return mM, fn, tvals, ons, xy, BL, sbl



def ind_close_to_point(xy, As, point):
    mytree = spatial.cKDTree(As)
    dist, indexes = mytree.query(point)
    
    mytree = spatial.cKDTree(xy)
    dist, indexes = mytree.query(As[indexes])
    
    return indexes
    
    

if __name__ == '__main__':

    input_dir = '/Users/lisa/Dropbox/Research/2016_paper_figures/Design_figure/Useable/'#'/Users/lisa/Dropbox/Research/2016_paper_figures/Design_figure/same_points_diff_coordination_3pts_1bond_del/'
    output_dir = input_dir + 'strip_data2/'
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    
    fns = glob.glob(input_dir + '*.pickle')
input_dir = '/Users/lisa/Dropbox/Research/2016_paper_figures/Design_figure/same_points_diff_coordination_4pts_3bond_delr/'
output_dir = input_dir + 'strip_data/'

input_dir = '/Users/lisa/Dropbox/Research/2016_paper_figures/Design_figure/Useable/'#'/Users/lisa/Dropbox/Research/2016_paper_figures/Design_figure/same_points_diff_coordination_3pts_1bond_del/'
output_dir = input_dir + 'strip_data2/'
if not os.path.exists(output_dir): os.mkdir(output_dir)

fns =glob.glob(input_dir + '*.pickle')
#['/Users/lisa/Dropbox/Research/2016_paper_figures/Design_figure/new_five_points_1/data_dict_007_fconfig.pickle']#glob.glob(input_dir + '*.pickle')
for k in range(len(fns)):
    #k = 4
    fn = fns[k]

    new_fn = fn.split('/')[-1]
    new_fn = new_fn.split('.')[0]
    print 'new fn is', new_fn

    of = open(fn, 'rb')
    data = pickle.load(of)

    R = data['R']
    xy = data['xy']
    BL = data['BL']
    a = data['a']
    ons = data['ons']
    sbl = [i%len(R) for i in range(len(xy))]

    A = R[0]
    B = R[1]
    C = R[2]


    xy, BL, sbl = automatic_semiinfinite_cell(3, 25, xy, BL, np.array(sbl), a, check = False, get_BL = False)
    del_points = remove_ext(xy, BL)



    while len(del_points)>0:
        xy, BL, sbl = delete_point(xy, BL, sbl, del_points[0])
        del_points = remove_ext(xy, BL)
        
    mM, new_fn_2, tvals, ons1, xy, BL, sbl = calc_semi_in(R, xy, BL, sbl, ons)
    Rp, NiP, NkP, cols, line_cols, vertex_points = make_vis(xy, a, BL, np.array(sbl))


    fig = sps.figure_in_mm(150,150)
    ax = fig.gca()

    lf.lattice_plot(Rp, NiP, NkP, ax,cols, line_cols)
    patch = [Polygon(vertex_points)]
    p = PatchCollection(patch, alpha = 0.5, zorder= 0)

    ax.add_collection(p)
    ax.set_aspect(1)


    #plt.scatter(Rp[in_cell, 0], Rp[in_cell, 1], c= 'w')     
    #plt.scatter(xy2[:, 0], xy2[:, 1], c= 'w')
    ax.axis('off')
    plt.savefig(output_dir + '_' + new_fn +'.pdf')
    plt.show()
    plt.close()
        # 
        # 
        # if not os.path.isfile(output_dir + new_fn +'_strip'):
        #     kx = np.arange(-2., 2., 0.05)
        #     ky = np.zeros_like(kx)
        #     eigs = []
        #     eigv = []
        #     for j in range(len(kx)):
        #         if j%10==0:
        #             print j
        #         matri = mM([kx[j], ky[j]])
        #        
        #         eigval, eigvect = np.linalg.eig(matri)
        #         eigval = np.real(eigval)
        #         si = np.argsort(eigval)
        #         eigvect = eigvect[:,si]
        #         eigval = eigval[si]
        #         eigs.append(eigval)
        #         eigv.append(eigvect)
        #     
        #     bands = np.array(eigs)
        # 
        #     dat_dict = {'tvals':tvals, 'ons':ons1, 'kx':kx, 'ky':ky , 'bands':eigs, 'eigv':eigv}
        #     print new_fn
        #     
        #     cf.save_pickled_data(output_dir,  new_fn + '_strip', dat_dict)
        # else:
        #     of = open(output_dir + new_fn +'_strip', 'rb')
        #     data = pickle.load(of)
        #     ky = data['ky']
        #     kx = data['kx']
        #     bands = data['bands']
        #     eigv = data['eigv']
        #     
        # bands = np.array(bands)
        # 
        # fig = plt.figure()
        # for i in range(len(bands[0])):
        #     plt.plot(kx, bands[:,i], c= 'k')
        # plt.savefig(output_dir+new_fn+'_strip.png')