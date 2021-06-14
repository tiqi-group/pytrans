#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 01-2021 - Carmelo Mordini <carmelo> <cmordini@phys.ethz.ch>
"""
Module docstring

"""
import numpy as np
import matplotlib.pyplot as plt

from pytrans.utils.cryo_analysis import analyse_pot

from pytrans.constants import um, MHz

from pytrans.trap_model.cryo import CryoTrap as Trap
from pytrans.potential_well import PotentialWell
from pytrans.waveform_solver import Solver


x0 = 250 * um
depth = 0.05
axial = 1 * MHz
split = 1.2 * MHz
tilt = 30  # degrees

trap = Trap()
well0 = PotentialWell(0, depth, axial, split, tilt)
well = PotentialWell(x0, depth, axial, split, tilt)
wells = [well0, well]
solver = Solver(trap, wells)

voltages = solver.static_solver(r0=0)
voltages = voltages.value[0]
print(voltages)

args = (
    np.asarray((x0, 0, trap.z0)),
    trap.electrode_indices,
    trap.Vrf,
    trap.Omega_rf
)

analyse_pot(voltages, *args)
x = trap.transport_axis
moments = trap.eval_moments(x)
pot = np.sum([well.gaussian_potential(x) for well in wells], axis=0)

plt.figure()
plt.plot(x, pot)
plt.plot(x, voltages @ moments)

plt.figure()
plt.plot(voltages)

plt.show()
