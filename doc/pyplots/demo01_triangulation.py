import numpy as np
import lattice_elasticity as le
import matplotlib.pyplot as plt

'''Demonstrate the functions that create lattices from a point set via triangulation.
These can identify boundaries, remove triangulation artifacts, and tune the coordination of the lattice'''

xy = np.random.randn(17, 2)
# Show what just a simple triangulation would do:
NL,KL,BL,BM = le.delaunay_lattice_from_pts(xy,trimbound=False)
TRI = le.BL2TRI(BL)
fig = plt.figure()
plt.gca().set_aspect('equal')
plt.triplot(xy[:,0], xy[:,1], TRI, 'bo-')

# Show how to kill unnatural boundaries (trimbound)
NL,KL,BL,BM = le.delaunay_lattice_from_pts(xy,trimbound=True, thres=10.0)
TRI = le.BL2TRI(BL)
fig = plt.figure()
plt.gca().set_aspect('equal')
plt.triplot(xy[:,0], xy[:,1], TRI, 'go-')

# Extract the boundary of a bonded point set
boundary = le.extract_boundary_from_NL(xy,NL,KL)
    
# Plot the boundary to show its extraction
fig = plt.figure()
plt.clf()
ax = plt.axes()
ax.set_aspect('equal')
ax.plot(xy[:,0],xy[:,1],'b.')
#lines = [zip(xy[boundary[i],0], xy[boundary[i],1]) for i in range(len(BL))]
ax.plot(xy[boundary,0],xy[boundary,1],'b-')
ax.plot(xy[[boundary[-1],boundary[0]],0],xy[[boundary[-1],boundary[0]],1],'b-')


# Make BL to make TRI
BL = le.NL2BL(NL,KL)
TRI = le.BL2TRI(BL)

# Show how to identify the boundary triangles
btri = le.boundary_triangles(TRI,boundary)
#print 'Identified the boundary triangles as:', TRI[btri]
zfaces = np.zeros(len(TRI),dtype=float)
zfaces[btri] = 1.
# Color the boundary triangles in a plot
plt.figure()
plt.gca().set_aspect('equal')
plt.tripcolor(xy[:,0], xy[:,1], TRI, facecolors=zfaces, edgecolors='k')
plt.colorbar()
plt.title('tripcolor() showing boundary tris')    