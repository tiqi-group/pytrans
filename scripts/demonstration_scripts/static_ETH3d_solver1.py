#!/usr/bin/env python3
# static with ETH3d using solver1
# author: Sebastian Heinekamp

# imports
from HOA2Trap import HOA2Trap as HOA2Moments
from ETH3dTrap import ETH3dTrap as ETHMoments
import loading_utils as lu
import transport_utils as tu
import copy as cp
from reorder import *
from pytrans import *
import numpy

import sys
sys.path.append("../")

# Loading conveyor stuff

# available Traps


# Expremient settings
pos = -0.0004
freq = 2000000
offset = 1


weights = [20000, 18000, 500, 20000, 500, 100, 0, 200, 0, 0]  # default weights used in the SolvePotentials2 routine (betas)
# solver2_weights=[   0,18000, 500,20000, 500,100,0,200,0,1e6], # offset relaxation
# solver2_weights=[50000,18000,1000,20000,1000,100,0, 20,0,0], #  HOA2 weights
# d_full=30e-3, #Mainz Voltag restriction off
# d_part=40e-3, #Mainz voltage restriction off
# d_full=10e-2,
# d_part=20e-2,
dfull = 3.8e-3
dpart = 7e-3
# d_full=5e-3,
# d_part=9e-3,

# print(global_settings['USESOLVE2'])
global_settings['USESOLVE2'] = False

trap = ETHMoments()
trap_mom.overwriteGlobalVariables()  # makes sure the global variables correspond to the trap variables - only needed if not executed in Moments constructor

wdw2 = WavDesiredWells([pos], [freq], [offset], trap_m=trap, solver2_weights=weights, d_full=dfull, d_part=dpart)
wf = Waveform(wdw2)

mu2 = wf.raw_samples()
# print(mu2)

# calculacte errors
a = 2 * (2 * np.pi * freq)**2 * (mass_Ca * atomic_mass_unit) / (2 * electron_charge)
print("The offset error (normalized): ")
print((numpy.dot(mu2.T, trap.Func(pos, 0)) - offset) / offset)
print("The first derivative at the well bottom")
print(numpy.dot(mu2.T, trap.Func(pos, 1)))
print("The frequency error (normalized): ")
print((numpy.dot(mu2.T, trap.Func(pos, 2)) - a) / a)
