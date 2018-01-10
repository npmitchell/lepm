import numpy as np
import lattice_elasticity as le
import matplotlib.pyplot as plt

xy = np.random.randn(50, 2)
# Show what just a simple triangulation would do:
NL,KL,BL,BM = le.delaunay_lattice_from_pts(xy,trimbound=False)
TRI = le.BL2TRI(BL,xy)
fig = plt.figure()
plt.gca().set_aspect('equal')
plt.triplot(xy[:,0], xy[:,1], TRI, 'bo-')
plt.show()

# Now tune coordination, after automatically computing the starting coordination
NL,KL,BL,BM = le.delaunay_lattice_from_pts(xy,trimbound=True,target_z=2.5,zmethod='highest')
TRI = le.BL2TRI(BL,xy)
fig = plt.figure()
plt.gca().set_aspect('equal')
#plt.triplot(xy[:,0], xy[:,1], TRI, 'bo-')
le.display_lattice_2D(xy,BL,title='',close=False)
for i in range(len(xy)):
    plt.text(xy[i,0]+0.05,xy[i,1],str(i))
#print 'NL = ', NL
#print 'KL = ', KL
#print 'avg(z) = ', float(np.count_nonzero(KL))/float(len(KL))
plt.show()