# Info on model files for the Cryo setup

root = '/scratch/TrapSimulation_Karan' on tiqibacon

All csv files here have been exported by Karan using RunOneElec.m and RunAllElecs.m, which run `2018_12_05_Axial2qubitTrapSim_DC.mph`

The `{x, y, z}line` files export

## Axial2QubitTrap_xline_DC[1..10].csv

These should be the moments of the DC electrodes, labelled from D1 to D10

- what about the RF electrodes?
- what about the two central DC electrodes running across x? Are they the 'shims'?
- wtf is x0?

There are also files for the electric fields and 2d data, I'll learn how to use them later

**NB** In COMSOL, the reference frame is rotated 180 degrees around z wrt the lab (and the article)'s frame!

---
Carmelo
