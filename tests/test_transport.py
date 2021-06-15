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

samples = 111
x0 = np.linspace(0, x0, samples)

trap = Trap()
wells = [
    PotentialWell(x0, depth, axial, split, tilt, freq_pseudo=trap.freq_pseudo, scale_roi=1),
]
n_wells = len(wells)
solver = Solver(trap, wells)

# vw0 = np.asarray([0.00317428, - 0.02277012, 0.08087201, - 0.1745012, 0.63677583, - 1.30254116,
#                   0.63677459, - 0.17449671, 0.08086666, - 0.02276043, 0.00317427, - 0.02277012,
#                   0.08087201, - 0.1745012, 0.63677583, - 1.30254116, 0.63677459, - 0.17449671,
#                   0.08086666, - 0.02276043, ]).reshape(samples, -1)


voltages = solver.solver(rx=1, rh=0, r0=0,
                         rd=1e3,
                         method_x='g',
                         verbose=True)

a, b = 0, -2
vv0 = voltages.value[a]
vv1 = voltages.value[b]
vvs = [vv0, vv1]
poss = x0[a], x0[b]

args = (
    trap.electrode_indices,
    trap.Vrf,
    trap.Omega_rf
)

# fig, _axes = plot3d_make_layout(2, squeeze=False)
# for j, vv in enumerate(vvs):
#     print(f"\nsample{j}:")
#     analyse_pot(vv, np.asarray((poss[j], 0, trap.z0)), *args, axes=_axes[j])

x = trap.transport_axis
moments = trap.eval_moments(x)


fig, (ax, ax1) = plt.subplots(1, 2, figsize=(12, 4))

for j, vv in enumerate(vvs):
    pot = np.sum([well.gaussian_potential(x, j) for well in wells], axis=0)
    l, = ax.plot(x * 1e6, pot, '--')
    ax.plot(x * 1e6, vv @ moments, '-', color=l.get_color())

    ax1.plot(vv)

plot_electrodes(ax, scale=1)

fig, (ax, ax1) = plt.subplots(1, 2, figsize=(12, 4))
pp = voltages.value @ moments
ax.imshow(pp)
ax.set_aspect('auto')

w = wells[0]
ppw = np.stack([w.gaussian_potential(x, j) for j in range(w.samples)], axis=0)
ppw = ppw - ppw.mean(axis=1).reshape(-1, 1)
ax1.imshow(ppw)
ax1.set_aspect('auto')

plt.show()
