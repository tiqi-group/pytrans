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


x0 = 250 * um
depth = 0.05
axial = 1.34 * MHz
split = 1 * MHz
tilt = 30  # degrees

trap = Trap()
wells = [
    PotentialWell(37e-6, depth, axial, split, tilt, freq_pseudo=trap.freq_pseudo, scale_roi=1),
    # PotentialWell(x0, depth, axial, split, tilt, freq_pseudo=trap.freq_pseudo, scale_roi=1)
]
n_wells = len(wells)
solver = Solver(trap, wells)

vw0 = np.asarray([0.00317428, - 0.02277012, 0.08087201, - 0.1745012, 0.63677583, - 1.30254116,
                  0.63677459, - 0.17449671, 0.08086666, - 0.02276043, 0.00317427, - 0.02277012,
                  0.08087201, - 0.1745012, 0.63677583, - 1.30254116, 0.63677459, - 0.17449671,
                  0.08086666, - 0.02276043, ]).reshape(1, -1)

solver.uopt.value = vw0

voltages = solver.static_solver(rx=1, rh=0, r0=0,
                                method_x='g',
                                verbose=False)
voltages = voltages.value[0]
print(voltages)

args = (
    trap.electrode_indices,
    trap.Vrf,
    trap.Omega_rf
)

fig, _axes = plot3d_make_layout(n_wells, squeeze=False)
for j, w in enumerate(wells):
    print(f"\nwell{j}:")
    analyse_pot(voltages, np.asarray((w.x0[0], 0, trap.z0)), *args, axes=_axes[j])

    # plot_3dpot(tot_potential_ps, np.asarray((w.x0[0], 0, trap.z0)), args=(voltages,) + args, roi=(400, 30, 30), axes=_axes[j])
    # plot_3dpot(tot_potential_ps, np.asarray((w.x0[0], 0, trap.z0)), args=(vw0[0],) + args, roi=(400, 30, 30), axes=_axes[j])

x = trap.transport_axis
moments = trap.eval_moments(x)
pot = np.sum([well.gaussian_potential(x) for well in wells], axis=0)

fig, (ax, ax1) = plt.subplots(1, 2, figsize=(12, 4))
ax.plot(x * 1e6, pot)
ax.plot(x * 1e6, voltages @ moments)
plot_electrodes(ax, scale=1)

ax1.plot(voltages)
ax1.plot(vw0[0])

plt.show()
