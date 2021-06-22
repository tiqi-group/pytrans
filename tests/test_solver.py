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
from pytrans.utils.cryo_plotting import plot_electrodes, plot3d_make_layout

from pytrans.constants import MHz

from pytrans.trap_model.cryo import CryoTrap as Trap
from pytrans.potential_well import PotentialWell
from pytrans.solver import Solver

x0 = 0
depth = 0.1
axial = 1.3 * MHz
split = -2 * MHz
tilt = -3 * MHz

trap = Trap()
wells = [
    PotentialWell(x0, depth, axial, split, tilt, freq_pseudo=trap.freq_pseudo, scale_roi=1),
    # PotentialWell(250e-6, depth, axial, split, tilt, freq_pseudo=trap.freq_pseudo, scale_roi=1),
]
n_wells = len(wells)
solver = Solver(trap, wells)


voltages = solver.solver(rx=1, rh=0.1, r0=0.,
                         method_x='g',
                         verbose=False)

# voltages = solver.solver(rx=1, rh=0.1, r0=0.1,
#                          method_x='q',
#                          verbose=False)

voltages = voltages.value[0]
print(voltages)

v_calc = trap.calculate_voltage(axial, split, tilt)
print(v_calc)

args = (
    trap.electrode_indices,
    trap.Vrf,
    trap.Omega_rf
)

fig, axess = plot3d_make_layout(2, squeeze=False)
for j, w in enumerate(wells):
    print(f"\nwell{j}:")
    analyse_pot(voltages, np.asarray((w.x0[0], 0, trap.z0)), *args, axes=axess[j])
    axess[j][3].set_title(f'well {j}')

print("\ncalculate_voltage:")
analyse_pot(v_calc, np.asarray((x0, 0, trap.z0)), *args, axes=None)  # , roi=(400, 30, (-30, 100)))

x = trap.transport_axis
moments = trap.eval_moments(x)

fig, (ax, ax1) = plt.subplots(1, 2, figsize=(12, 4))
ax.plot(x * 1e6, voltages @ moments)
ax.plot(x * 1e6, v_calc @ moments)
plot_electrodes(ax, scale=1)

ax1.plot(voltages)
ax1.plot(v_calc)

plt.show()
