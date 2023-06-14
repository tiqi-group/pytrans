# List of breaking changes in go_public

## from v.2.1.0

- Trap models are now based on Electrodes
- Change AbstractTrap to AbstractTrapModel
- Change signatures of various functions to explicitly pass ion species (ion_mass). This includes:
  - Model methods
  - Objectives

## TODO

- plotting.py: rename `plot3d_` methods to `plot_potential_` to introduce real 3d plot
