#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 08/2021
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

'''
Module docstring
'''

import numpy as np
import matplotlib.pyplot as plt

from pytrans.trap_model.cryo import CryoTrap
from pytrans.potential_well import freq_to_curv
from pytrans.constants import ion_masses, elementary_charge
from pytrans.utils.linspace_functions import zpspace
from pytrans import objectives as obj
from pytrans.solver import solver

from pytrans.utils.cryo_analysis import analyse_pot
from pytrans.utils.cryo_plotting import plot_electrodes, plot3d_make_layout

from pprint import pprint


# selected_electrodes = [0, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 19]
selected_electrodes = list(range(20))

trap = CryoTrap(selected_electrodes=selected_electrodes)

xx = zpspace(0, 375e-6, 256)  # [0.]
axial = 1.9e6
tilt = -5e6
curvature = freq_to_curv(axial, ion_masses['Ca'], elementary_charge)

v_calc2, _ = trap.from_static_params(axial, 0, tilt, 2.55, 0.25, 4.12, center=6)
v_calc3, _ = trap.from_static_params(axial, 0, tilt, center=9)


def weight_per_electrode(x, s=100e-6):
    w = 1 - np.exp(-x**2 / 2 / s**2)
    w[[0, 7]] = 0
    return w


step_objectives = [
    [obj.VoltageObjective(v_calc2[selected_electrodes], constraint_type='==')]
]

step_objectives += [
    [
        obj.PotentialObjective(_x, derivatives=1, value=0),
        obj.PotentialObjective(_x, derivatives='xx', value=curvature),
        obj.PotentialObjective(_x, derivatives=['xy', 'xz'], value=0),
        obj.PotentialObjective(_x, derivatives='yz', value=1e8, constraint_type='>='),
        obj.PotentialObjective(_x, derivatives=['yy', 'zz'], value=0, constraint_type='>='),
        # obj.VoltageObjective(0, weight=1e5, voltage_weights=weight_per_electrode(trap.electrode_x - _x))
    ] for _x in xx[1:]
]

step_objectives[-1] = [obj.VoltageObjective(v_calc3[selected_electrodes], constraint_type='==')]


global_objectives = [
    obj.SlewRateObjective(weight=1e6),
    obj.VoltageObjective(10, constraint_type='<='),
    obj.VoltageObjective(-10, constraint_type='>=')
]

waveform, final_costs = solver(trap, step_objectives, global_objectives, verbose=True)
print(waveform[0].shape, waveform[0].value)
pprint(final_costs)

trap.generate_waveform(waveform.value, index=0, description='2 -> 3',
                       waveform_filename='transport_2_to_3.dwc.json',
                       monitor_values=xx * 1e4,
                       verbose=True)


# ------------- analysis

x = trap.x

args = (
    trap._electrode_indices,
    trap.Vrf,
    trap.Omega_rf
)

to_analyse = {}


def _add(name, j):
    vv = np.zeros((20,))
    vv[selected_electrodes] = waveform[j].value
    to_analyse[name] = (vv, xx[j])


_add('w_start', 0)
if len(xx) > 1:
    _add('w1', len(xx) // 3)
    _add('w2', 2 * len(xx) // 3)
    _add('w_stop', -1)

n = len(to_analyse)
fig0, axess = plot3d_make_layout(n, squeeze=False)
fig, (ax, ax1) = plt.subplots(1, 2, figsize=(12, 4))

for j, (name, (vs, x0)) in enumerate(to_analyse.items()):
    print('\n' + name)
    axx = axess[j]
    analyse_pot(vs, np.asarray((x0, 0, trap.z0)), *args, axes=axx)
    axx[3].set_title(name)

    ax.plot(x * 1e6, trap.potential(vs), label=name)
    ax1.plot(vs)

plot_electrodes(ax, scale=1)

fig, ax = plt.subplots()
for j, v_electrode in zip(trap.electrode_indices, waveform.value.T):
    ax.plot(v_electrode, label=f"Ele{j:02d}")
ax.legend()

fig, (ax, ax1) = plt.subplots(1, 2, figsize=(12, 4))
im = waveform.value @ trap.moments
for _x, v in zip(xx, im):
    offset = v[np.argmin(abs(x - _x))]
    v -= offset
ax.imshow(im)
ax.set_aspect('auto')

l, = ax1.plot(x * 1e6, im[0])
ax1.set_ylim(im.min() - 0.05 * im.ptp(), im.max() + 0.05 * im.ptp())

for j in range(len(im)):
    l.set_ydata(im[j])
    plt.pause(0.02)

plt.show()
