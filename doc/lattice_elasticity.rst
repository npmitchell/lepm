lattice_elasticity
=====================

Module with auxiliary functions for creating and evolving lattices (of springs, masses, gyros)

Common Variables
----------------
Some common definitions of variables used throughout are:

xy : N x 2 float array
    2D positions of points (positions x,y). Row i is the x,y position of the ith particle.
NL : array of dimension #pts x max(#neighbors)
    The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
    maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
KL : NP x max(#neighbors) int array
    spring connection/constant list, where 1 corresponds to a true connection,
    0 signifies that there is not a connection, -1 signifies periodic bond
BM : array of length #pts x max(#neighbors)
    The (i,j)th element is the bond length of the bond connecting the ith particle to its jth neighbor (the particle with index NL[i,j]).
BL : array of dimension #bonds x 2
    Each row is a bond and contains indices of connected points. Negative values denote particles connected through periodic bonds.
bL : array of length #bonds 
    The ith element is the length of of the ith bond in BL
LL : tuple of 2 floats
    Horizontal and vertical extent of the bounding box (a rectangle) through which there are periodic boundaries.
    These give the dimensions of the network in x and y, for S(k) measurements and other periodic things.
BBox : #vertices x 2 float array
    bounding polygon for the network, usually a rectangle
eigval : 2*N x 1 complex array
    eigenvalues of the matrix, sorted by order of imaginary components
eigvect : typically 2*N x 2*N complex array
    eigenvectors of the matrix, sorted by order of imaginary components of eigvals
    Eigvect is stored as NModes x NP*2 array, with x and y components alternating, like:
    x0, y0, x1, y1, ... xNP, yNP.
polygons : list of int lists
    indices of xy points defining polygons.
NLNNN : array of length #pts x max(#next-nearest-neighbors)
    Next-nearest-neighbor array: The ith row contains indices for the next nearest neighbors for the ith point.
KLNNN : array of length #pts x max(#next-nearest-neighbors)
    Next-nearest-neighbor connectivity/orientation array:
    The ith row states whether a next nearest neighbors is counterclockwise (1) or clockwise(-1)
PVxydict : dict
    dictionary of periodic bonds (keys) to periodic vectors (values)
    If key = (i,j) and val = np.array([ 5.0,2.0]), then particle i sees particle j at xy[j]+val
    --> transforms into:  ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen by particle i
PVx : NP x NN float array (optional, for periodic lattices)
    ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen by particle i
PVy : NP x NN float array (optional, for periodic lattices)
    ijth element of PVy is the y-component of the vector taking NL[i,j] to its image as seen by particle i


Example Usage
-------------
.. currentmodule:: lattice_elasticity

Demonstrating lattice construction from point set:   

.. plot:: pyplots/demo01_triangulation.py
    :include-source:
.. plot:: pyplots/demo02_coordination.py
    :include-source:
.. plot:: pyplots/demo03_polygons.py
    :include-source:
    
Overview
--------
.. currentmodule:: lattice_elasticity

.. autosummary::

Classes and Functions
---------------------
   
.. automodule:: lattice_elasticity
    :members: 
    :show-inheritance:
