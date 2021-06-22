#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 01-2021 - Carmelo Mordini <carmelo> <cmordini@phys.ethz.ch>
"""
Module docstring

"""
import numpy as np
import matplotlib.pyplot as plt

from pytrans.utils.cryo_analysis import analyse_pot, tot_potential_ps
from pytrans.utils.cryo_plotting import plot_electrodes, plot3d_make_layout, plot_3dpot

from pytrans.constants import um, MHz

from pytrans.trap_model.cryo import CryoTrap as Trap
from pytrans.potential_well import PotentialWell
from pytrans.solver import Solver

x0 = 0
depth = 0.05
axial = 1.9
split = -2
tilt = -3

trap = Trap()
wells = [
    PotentialWell(x0, depth, axial, split, tilt, freq_pseudo=trap.freq_pseudo, scale_roi=1),
]
n_wells = len(wells)
solver = Solver(trap, wells)


voltages = solver.solver(rx=0, rh=1, r0=0,
                         method_x='g',
                         verbose=False)
voltages = voltages.value[0]
print(voltages)

v_static = trap.calculate_voltage(axial, split, tilt)[:20]
print(v_static)

args = (
    trap.electrode_indices,
    trap.Vrf,
    trap.Omega_rf
)

fig, (axes1, axes2) = plot3d_make_layout(2, squeeze=False)
print("solver:")
analyse_pot(voltages, np.asarray((x0, 0, trap.z0)), *args, axes=axes1)

print("static:")
analyse_pot(v_static, np.asarray((x0, 0, trap.z0)), *args, axes=axes2)

x = trap.transport_axis
moments = trap.eval_moments(x)
pot = np.sum([well.gaussian_potential(x) for well in wells], axis=0)

fig, (ax, ax1) = plt.subplots(1, 2, figsize=(12, 4))
ax.plot(x * 1e6, pot)
ax.plot(x * 1e6, voltages @ moments)
plot_electrodes(ax, scale=1)

ax1.plot(voltages)
ax1.plot(v_static)

plt.show()
