# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 14:25:57 2016

@author: Robin Oswald

This script imports the results of the trap NIST BEM simulations (e.g. trap_exp.vtk)
in the vtk format, and exports the relevant data in python convenient data structures for
easy import later on.
"""

import pyvtk
# Notes on using pyvtk in python3.5 to read trap BEM files:
# For pyvtk to successfully read trap_exp.vtk, had to do these changes:
# 1) Reverse order in which 'spacing' and 'origin' are read
# 2) add a .decode() call to a readline call due to byte<>str type error
import numpy as np
import pickle

# %% Read vtk file and save contents in internal data structures
trapvtk = pyvtk.VtkData('trap_exp.vtk')

# Electrode potentials and RF field
electrode_potentials = {}
for dataset in trapvtk.point_data.data:
    if isinstance(dataset, pyvtk.Vectors): # Field (RF)
        RF_field = np.array(dataset.vectors)
    elif isinstance(dataset, pyvtk.Scalars): # Potentials (DC or RF)
        electrode_potentials.update( {dataset.name : np.array(dataset.scalars)} )
        
# Coordinate grid
origin = np.array(trapvtk.structure.origin)
spacing = np.array(trapvtk.structure.spacing)
dimensions = np.array(trapvtk.structure.dimensions)

x = origin[0] + spacing[0] * np.arange(dimensions[0])
y = origin[1] + spacing[1] * np.arange(dimensions[1])
z = origin[2] + spacing[2] * np.arange(dimensions[2])

yy,xx,zz = np.meshgrid(y,x,z) # Swapping x and y is required (see below)
xx = xx.flatten(order='F') # order='F' required
yy = yy.flatten(order='F') # These choices ensure that we are consistent with
zz = zz.flatten(order='F') # the ordering of the points implicit in the vtk file

coordinates = np.column_stack( ( xx, yy, zz) )

# %% Save to trap_exp.pickle

with open('trap_exp.pickle', 'wb') as f:
    pickle.dump( [electrode_potentials, origin, spacing, dimensions, x, y, z, xx, yy, zz, coordinates], f, pickle.HIGHEST_PROTOCOL)
    
# %% Load from trap_exp.pickle

# Minimum sample code demonstrating how to import the data again (to be used
# in other files that need to access this data).
if False:
    import numpy as np
    import pickle
    
    with open('trap_exp.pickle', 'rb') as f:
        electrode_potentials, origin, spacing, dimensions, x, y, z, xx, yy, zz, coordinates = pickle.load(f)