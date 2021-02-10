import numpy as np
from lepm import lattice_class
from lepm import gyro_lattice_class

geomList = ['iscentroid']

for geom in geomList:
    # Periodic
    subprocess.call(['python', 'gyro_lattice_class.py', '-LT', geom, '-periodic', '-NP', 512, '-nice_plot'])


nice_plot (lattice_class)
nice_plot (gyro_lattice_class)
redo_gxy
localization
save_ipr