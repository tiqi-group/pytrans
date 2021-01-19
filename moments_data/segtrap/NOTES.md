# Info on model files for the Cryo setup

## DanielTrapMomentsTransport.mat

DanielTrapMomentsTransport.mat was created based on the results from the trap simulations performed with the NIST BEM software.
Specifically, it is based on trap.vtk and was generated with create_moments_file.m

Of course python knows how to read matlab files:

```python
data = scipy.io.loadmat(moments_path, struct_as_record=False)['DATA'][0][0]
```

Here you have the following namespace:

- data.amu = 40. The mass of an ion
- data.w_t = 90. IDK
- data.transport_axis: array (943, 1) with trap axis coordinates [in meters]
- data.electrode: array (1, 50) with data for the 50 electrodes. The class also names them:
  - 0 to 14:  DCCc0..14
  - 15 to 29: DCCa0..14
  - 30 to 49: DCS[a..d][1..5]. But these are never used. They shim potentials are also read from the .pickle for 3d data.
  - All of them are mat_structs with a single name, moments: array (943, 6) with the potential [0] and its derivatives along x [1..5] [in V/m^n]
- data.RF_pondpot: array (943, 1) Potential due to RF electrodes along trap axis. Needs to be scaled with rf freq, voltage and ion mass.
- data.RF_V = 325. I assume this is a voltage on the RF electrodes. Go figure it out.

This I needed to figure it out myself reading the comments in (what now is) the segtrap TrapModel class, in `vn/solver_edit:pytrans.py` and plotting the data.

Please, always document your code.

## trap.vtk, trap.pickle

I guess they are the same of trap_exp.*, just someone run again the simulation with different parameters. Ask Daniel or Francesco for that.

Actually the difference is in this comment

```python
potential_path=os.path.join(os.path.dirname(__file__), "moments_file", "trap.pickle"),  # +- 1000um in axial, +-4um in radial direction
# potential_path = os.path.join(os.path.dirname(__file__), "moments_file", "trap_exp.pickle"), # +-100um in axial, +-11um in radial direction
```

trap.pickle is always opened like that:

```python
with open(potential_path, 'rb') as f:
    potentials, origin, spacing, dimensions, x, y, z, xx, yy, zz, coordinates = pickle.load(f)
```

This file is created by `vtk_to_pickle.py`, which is described on the wiki. It contains:

- potentials: this is a huge dictionary, read it later.

----
Carmelo

As for the following: I'd prefer that you wrote what that file contains, instead of how it has been generated. Now I have to go and read all of your scripts.
Thanks.

## trap_exp.vtk

trap_exp.vtk was created using the NIST BEM trap simulation software.

## trap_exp.mat

trap_exp.mat was created based on the results from the trap simulations performed with the NIST BEM software.
Specifically, it is based on trap_exp.vtk and was generated with readVTK_mod.m.

## trap_exp.pickle

trap_exp.pickle was created based on the results from the trap simulations performed with the NIST BEM software.
Specifically, it is based on trap_exp.vtk and was generated with readVTK_mod.m.

## Resources
NIST BEM Software: J:\Software\BEM
Paraview (visualize results): J:\Software\BEM

Matlab files: \wav_gen\bem_to_potentials, \wav_gen\moments_file, ...

Daniel 3D trap simulations: J:\Projects\3D_Trap\trap_simulation\daniels_full_trap
Specifically, the logfile trap_exp.log shows the simulation parameters and trap_exp.vtk contains the results.


## Interpreting trap_exp simulation 3d data: Paraview, Matlab & Python
This comparison shows how to consistently handle the 3d data from the trap BEM simulations in Matlab & Python, using ParaView as the reference.

See trap_exp_plotting_comparison.png

### Paraview (left): (File: trap_exp.vtk)
Visualizes the potential due to electrode 'DCCa07', with a threshold of 0.14. Values below that are not shown. Note that the potential is highest towards (x=middle ie 0, y=positive, z= negative). This is our reference for how the results are supposed to look, because Paraview can natively read vtk files.

### Matlab (top right): (File; trap_exp.mat, generated with readVTK_mod.m based on trap_exp.vtk)
Using the code below to plot the same potential as above, using the same threshold of 0.14. Values below it are shown in dark blue. Note that the potential is also highest towards (x=middle, y=positive, z=negative). Thus the coordinates generated with meshgrid match the ones from Paraview. Note that in the call to the meshgrid function, X and Y are swapped from what we would naively expect!

```matlab
load trap_exp.mat
[Y,X,Z] = meshgrid(trap.y,trap.x,trap.z);
trap.coordinates = [X(:) Y(:) Z(:)];
electrode = 'DCCa07';
ind=find(strcmp(trap.idDCElectrodes,electrode));
pot_xyz = squeeze(trap.dataDCElectrodes(ind,:,:,:));
pot_xyz(pot_xyz<0.140)=0;
scatter3(trap.coordinates(:,1)/um,trap.coordinates(:,2)/um,trap.coordinates(:,3)/um,10,pot_xyz(:))
```

### Python (bottom right): (File; trap_exp.pickle, generated with vtk_to_pickle.py based on trap_exp.vtk)
Using the code below to plot the same potential as above, using the same threshold of 0.14. Note that the x and y axis directions are each reversed, which is why the plot looks different compared to ParaView & Matlab. However, the potential is also highest towards (x=middle ie 0, y =positive, z=negative). Thus the coordinates generated in vtk_to_pickle.py with meshgrid produce results that are consistent with Paraview & Matlab.

```python
# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

# %% Load data
with open('trap_exp.pickle', 'rb') as f:
    electrode_potentials, origin, spacing, dimensions, x, y, z, xx, yy, zz, coordinates = pickle.load(f)

# %% Plot data
um = 1e-6
pot = electrode_potentials['DCCa7']
pot = pot * (pot > 0.14)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(azim=45)
idk = ax.scatter(xx/um,yy/um,zz/um,c=pot, cmap = 'viridis')
plt.colorbar(idk, fraction=0.046, pad=0.04)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```
