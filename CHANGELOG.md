# List of breaking changes in go_public

- Trap models are now based on Electrodes
- Change AbstractTrap to AbstractTrapModel just to be annoying
- Change signatures of various functions to explicitly pass ion species (ion_mass). This includes:
  - Model methods
