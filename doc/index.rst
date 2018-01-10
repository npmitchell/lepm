.. lepm documentation master file, created by
   sphinx-quickstart on Mon Nov 30 12:00:01 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Lattice Elasticity Python Module documentation
============================================================

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
    The ith row states whether a next nearest neighbors is counterclockwise (1) or clockwise (-1)
PVxydict : dict
    dictionary of periodic bonds (keys) to periodic vectors (values)
    If key = (i,j) and val = np.array([ 5.0,2.0]), then particle i sees particle j at xy[j]+val
    --> transforms into:  ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen by particle i
PVx : NP x NN float array (optional, for periodic lattices)
    ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen by particle i
PVy : NP x NN float array (optional, for periodic lattices)
    ijth element of PVy is the y-component of the vector taking NL[i,j] to its image as seen by particle i
nljnnn : #pts x max(#NNN) int array
    nearest neighbor array matching NLNNN and KLNNN. nljnnn[i, j] gives the neighbor of i such that NLNNN[i, j] is
    the next nearest neighbor of i through the particle nljnnn[i, j]
kljnnn : #pts x max(#NNN) int array
    bond array describing periodicity of bonds matching NLNNN and KLNNN. kljnnn[i, j] describes the bond type
    (bulk -> +1, periodic --> -1) of bond connecting i to nljnnn[i, j]
klknnn : #pts x max(#NNN) int array
    bond array describing periodicity of bonds matching NLNNN and KLNNN. klknnn[i, j] describes the bond type
    (bulk -> +1, periodic --> -1) of bond connecting nljnnn[i, j] to NLNNN[i, j]
PV : 2 x 2 float array
    periodic lattice vectors, with x-dominant vector first, y-dominant vector second.
gxy : tuple of NX x NY arrays
    two-point correlation function in the positions of particles as function of vector distance x,y
gr :
    two-point correlation function in the positions of particles as function of scalar distance

Boundary conditions
-------------------
tug : give particles initial velocity
pull : displace particles continually with pullrate
offset : displace particles at boundaries
randomize : initially displace particles everywhere
fixed : displace particles at boundaries and keep fixed


Contents
--------

.. toctree::
    :maxdepth: 1
    :numbered:
    
    collapse_curves
    data_handling
    dataio
    gyro_collection
    gyro_lattice_class
    gyro_lattice_functions
    lattice_class
    lattice_collection
    lattice_elasticity
    lattice_functions
    le_geometry
    line_segments
    magnetic_gyro_functions
    magnetic_gyro_lattice_class
    mass_lattice_class
    meanfield_bond_energy
    penrose_tiling
    polygon_functions
    rename_files
    run_metaseries
    run_series
    script_checkNNNangles
    script_chern_kspace
    script_copy_movies
    script_edgelocalization_periodicstrip
    script_kitaevABsites
    script_semi_infinite_strip
    script_semi_infinite_strip_haldane
    script_summarize_decorations_figs
    stringformat
    structure
    twisty_lattice
    bianco_chern_class
    bianco_chern_functions
    build_cairo
    build_conformal
    build_dislocatedlattice
    build_hexagonal
    build_hucentroid
    build_hyperuniform
    build_iscentroid
    build_jammed
    build_kagcent_words
    build_kagcentframe
    build_kagome
    build_lattice_functions
    build_linear_lattices
    build_martini
    build_overcoordinated
    build_quasicrystal
    build_random
    build_randomspread
    build_randorg_gamma
    build_select_region
    build_square
    build_triangular
    build_voronoized
    make_lattice
    roipoly
    gammakick
    gammakick_calibration
    relax_pointset
    haldane_chern_class
    haldane_chern_collection
    haldane_chern_collection_collection
    haldane_chern_functions
    haldane_collection
    haldane_lattice_class
    haldane_lattice_functions
    kitaev_chern_class
    kitaev_chern_gyro_calc_finitesize_effect
    kitaev_chern_Haldane_calc
    kitaev_chern_Haldane_calc_finitesize_effect
    kitaev_chern_Haldane_calc_skeleton
    kitaev_collection
    kitaev_collection_collection
    kitaev_collection_functions
    kitaev_collection_nugrad_analysis_original
    kitaev_functions
    colormaps
    gyro_collection_plotting_functions
    gyro_lattice_plotting_functions
    haldane_chern_collection_plotting_functions
    haldane_chern_plotting_functions
    haldane_collection_plotting_functions
    haldane_lattice_plotting_functions
    isolum_rainbow
    kitaev_collection_plotting_functions
    kitaev_plotting_functions
    magnetic_gyro_movies
    movies
    network_visualization
    nplotstyle
    plotting
    plotting_haldane
    science_plot_style
    test_plot
    time_domain_ghst
    time_domain_gyros
    time_domain_magnetic
    time_domain_movies
    time_domain_plotting


To download and install the package on OS X, run the following commands
(*this will require your password to install, as well as a log-in to the Git
server*)::

    $ git clone ssh://<YOUR_NAME>@immenseirvine.uchicago.edu/git/noah
    $ cd noah/noah/lepm/
    $ sudo python setup.py develop
    
Updates can be installed running the following command from the ``noah`` directory::

    $ git pull

*If you have installed the module in develop mode, as above, you don't need
to re-run setup to use the new version.*
    
Once the package has been installed, all the sub-modules will be available
as members of the ``lepm`` package.  For example:

.. code-block:: python

    from lepm import lattice_elasticity  #module will appear as 'lattice_elasticity'
    import ilpm.lattice_elasticity       #module will appear as 'ilpm.lattice_elasticity'
    
    # Two ways of usage are then:
    rad_profile = lattice_elasticity.azimuthalAverage(image, center=None)
    rad_profile = ilpm.lattice_elasticity.azimuthalAverage(image, center=None)
    
Dependencies
============
There are several external modules used by LEPM:

    * Numpy/Scipy
    * Matplotlib
    * IrvineLab Python Modules (ILPM) (``ilpm.vector`` is used here and there)
    * `PyOpenCL <http://mathema.tician.de/software/pyopencl/>`_
      (used by ``simple_cl``, ``path``, and ``geometry_extractor``)
    * `Shapely <http://toblerity.org/shapely/manual.html>`_ (used in ``make_lattice``)
    * `descartes <https://bitbucket.org/sgillies/descartes>`_ (used by some functions in ``make_lattice.py``)

On OSX, PyOpenCL and descartes can be installed using ``easy_install``::

    $ sudo easy_install pyopencl
    $ sudo easy_install descartes 

Shapely can be installed using ``pip``::

    $ pip install Shapely

This *might* work on a Windows or Linux system, provided OpenCL and OpenGL are
present.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

