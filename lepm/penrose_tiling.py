import math
import cmath
# import cairo
import matplotlib.pyplot as plt
import matplotlib.path as mplpath
import numpy as np
import matplotlib.patches as mpatches
import lepm.lattice_elasticity as le


def generate_penrose_tiling(Num_sub, check=False):
    '''
    http://preshing.com/20110831/penrose-tiling-explained/
    '''
    # ------ Configuration --------
    NUM_SUBDIVISIONS = Num_sub
    # -----------------------------
    
    goldenRatio = (1 + math.sqrt(5)) / 2
    
    def subdivide(triangles):
        result = []
        for color, A, B, C in triangles:
            if color == 0:
                # Subdivide red triangle
                P = A + (B - A) / goldenRatio
                result += [(0, C, P, B), (1, P, C, A)]
            else:
                # Subdivide blue triangle
                Q = B + (A - B) / goldenRatio
                R = B + (C - B) / goldenRatio
                result += [(1, R, C, A), (1, Q, R, B), (0, R, Q, A)]
        return result
    
    def subdivide_kite_dart(triangles): 
        result = [] 
        for color, A, B, C in triangles: 
            if color == 0: 
                # Subdivide red (sharp isosceles) (half kite) triangle 
                Q = A + (B - A) / goldenRatio 
                R = B + (C - B) / goldenRatio 
                result += [(1, R, Q, B), (0, Q, A, R), (0, C, A, R)] 
            else: 
                # Subdivide blue (fat isosceles) (half dart) triangle 
                P = C + (A - C) / goldenRatio 
                result += [(1, B, P, A), (0, P, C, B)] 
        
        return result 
        
        # Create wheel of red triangles around the origin 
        triangles = [] 
        for i in xrange(10): 
            B = cmath.rect(1, (2*i - 1) * math.pi / 10) 
            C = cmath.rect(1, (2*i + 1) * math.pi / 10) 
            if i % 2 == 0: 
                B, C = C, B # Make sure to mirror every second triangle 
            triangles.append((0, B, 0j, C))
        
    def plot_quasicrystal(triangles):
        ax = plt.gca()
        for color, A, B, C in triangles:
            if color == 0:
                codes = [mplpath.Path.MOVETO,
                        mplpath.Path.LINETO,
                        mplpath.Path.LINETO,
                        mplpath.Path.CLOSEPOLY,
                        ]
                polygon = np.array([[A.real,A.imag], [B.real,B.imag], [C.real,C.imag], [A.real,A.imag]])
                path = mplpath.Path(polygon, codes)
                patch = mpatches.PathPatch(path, facecolor='orange', lw=2)
                ax.add_patch(patch)
                
            if color == 1:
                codes = [mplpath.Path.MOVETO,
                        mplpath.Path.LINETO,
                        mplpath.Path.LINETO,
                        mplpath.Path.CLOSEPOLY,
                        ]
                polygon = np.array([[A.real,A.imag], [B.real,B.imag], [C.real,C.imag], [A.real,A.imag]])
                path = mplpath.Path(polygon, codes)
                patch = mpatches.PathPatch(path, facecolor='blue', lw=2)
                ax.add_patch(patch)
        return ax
    
    # Create wheel of red triangles around the origin
    triangles = []
    for i in xrange(10):
        B = cmath.rect(1, (2*i - 1) * math.pi / 10)
        C = cmath.rect(1, (2*i + 1) * math.pi / 10)
        if i % 2 == 0:
            B, C = C, B  # Make sure to mirror every second triangle
        triangles.append((0, 0j, B, C))
    
    # Perform subdivisions
    for i in xrange(NUM_SUBDIVISIONS):
        triangles = subdivide(triangles)
    
    # Prepare cairo surface
    # surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, IMAGE_SIZE[0], IMAGE_SIZE[1])
    # cr = cairo.Context(surface)
    # cr.translate(IMAGE_SIZE[0] / 2.0, IMAGE_SIZE[1] / 2.0)
    # wheelRadius = 1.2 * math.sqrt((IMAGE_SIZE[0] / 2.0) ** 2 + (IMAGE_SIZE[1] / 2.0) ** 2)
    # cr.scale(wheelRadius, wheelRadius)
    
    # Draw red triangles
    plot_quasicrystal(triangles)
    plt.show()
    
    # Scale points
    mind = triangles[0][2]-triangles[0][1]
    scale = 1./mind
    tri = np.array(triangles)
    tri = tri[:,1:4]*scale
    plot_quasicrystal(triangles)
    
    # Convert points to numbered points
    # Create dict of locations to indices 
    indexd = {}
    xy = np.zeros((len(tri)*3,2))
    TRI = np.zeros_like(tri, dtype = int)
    rowIND = 0
    dmyi = 0
    offs = float(np.ceil(np.max(tri.real).ravel())+1)
    for AA, BB, CC in tri:
        # reformat A,B,C
        A = ('{0:0.2f}'.format(AA.real+offs),'{0:0.2f}'.format(AA.imag+offs))
        B = ('{0:0.2f}'.format(BB.real+offs),'{0:0.2f}'.format(BB.imag+offs))
        C = ('{0:0.2f}'.format(CC.real+offs),'{0:0.2f}'.format(CC.imag+offs))
        #print '\n\n\n'
        #print 'A = ', A
        if A not in indexd:
            indexd[A] = dmyi
            xy[dmyi] = [AA.real, AA.imag]
            TRI[rowIND,0] = dmyi
            dmyi += 1
            #print 'xy[0:dmyi,:] = ', xy[0:dmyi,:]
        else:
            index = indexd[A]
            TRI[rowIND,0] = index
        
        #print 'indexd = ', indexd
        #print '\nB = ', B
        if B not in indexd:
            indexd[B] = dmyi
            xy[dmyi] = [BB.real, BB.imag]
            TRI[rowIND,1] = dmyi
            dmyi += 1
            #print 'xy[0:dmyi,:] = ', xy[0:dmyi,:]
        else:
            index = indexd[B]
            TRI[rowIND,1] = index
        
        #print 'indexd = ', indexd
        #print '\nC = ', C
        if C not in indexd:
            indexd[C] = dmyi
            xy[dmyi] = [CC.real, CC.imag]
            TRI[rowIND,2] = dmyi
            dmyi += 1
            #print 'xy[0:dmyi,:] = ', xy[0:dmyi,:]
        else:
            index = indexd[C]
            TRI[rowIND,2] = index
        rowIND += 1
        
        #check
        #print 'indexd = ', indexd
        #print 'TRI[rowIND] = ', TRI[rowIND]
        #plt.plot(xy[:,0],xy[:,1],'b.')
        #plt.triplot(xy[:,0], xy[:,1], TRI, 'ro-')
        #for ii in range(len(xy)):
        #    plt.text(xy[ii,0],xy[ii,1], str(ii))
        #plt.pause(0.1)
    
    xy = xy[0:dmyi]
    print 'xy = ', xy
    
    #plot_quasicrystal(triangles)
    plt.plot(xy[:,0],xy[:,1],'b.')
    plt.triplot(xy[:,0], xy[:,1], TRI, 'ro-')
    plt.show()
    
    BL = le.TRI2BL(TRI)
    NL, KL = le.BL2NLandKL(BL, NP='auto', NN='min')
    print 'TRI = ', TRI
    print 'BL = ', BL
    if check:
        le.display_lattice_2D(xy,BL,close=False)
        for ii in range(len(xy)):
            plt.text(xy[ii,0],xy[ii,1], str(ii))
        plt.show()

    lattice_exten = 'penroserhomb_div_'+str(Num_sub)
    return xy, NL, KL, BL, lattice_exten


if __name__ == '__main__':
    generate_penrose_tiling(5, check=True)