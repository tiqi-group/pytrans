#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 01-2021 - Carmelo Mordini <carmelo> <cmordini@phys.ethz.ch>
"""
Module docstring

"""
import numpy as np
import matplotlib.pyplot as plt

from pytrans.waveform import Waveform
from pytrans.potential_well import curv_to_freq
from pytrans.constants import um, MHz
# from pytrans.solver import static_solver

from pytrans.trap_model.cryo import CryoTrap as Trap
# from pytrans.trap_model.segtrap import ETH3dTrap as Trap

from calculate_voltage import calculate_voltage

import lmfit

M = lmfit.models.QuadraticModel()
p0 = M.make_params(a=1, b=0, c=0)

pos = 250 * um
freq = 1 * MHz
offs = 0

energy_threshold = 0.1
trap = Trap()

wav = Waveform()
hw = wav.add_harmonic_well(freq, pos, offs, energy_threshold)
x = trap.transport_axis

roi = hw.roi(x)
x_roi = x[roi]
weights = hw.gaussian_weight(x)

# def reset_offset(v):
#     min = np.min(v[roi])
#     return v - min + offs


def pot_params(v, line):
    res = M.fit(v[roi], p0, x=x_roi, weights=weights)
    fitted = res.eval(x=x_roi)
    ax.plot(x_roi / um, fitted, ls='--', color=line.get_color())
    fr = curv_to_freq(res.params['a'].value)
    print(f"Pot {line.get_label()}: freq {fr / MHz:.3f} MHz")


# wav.add_harmonic_well(freq, 100 * um, offs, energy_threshold)

# uopt = static_solver(trap.moments, potential, roi, gaussian_weight, r0_weight=0, default_V=1)
uopt = wav.static_solver(trap, r0=1e-6, default_V=0)

fig, ax = plt.subplots()
for j, hw in enumerate(wav.harmonic_wells):
    # ax2 = ax.twinx()
    ax.plot(x_roi / um, hw.potential(x), label=f"target {j}")
    # ax2.plot(x_roi / um, hw.gaussian_weight(x), 'C0--')

upot = uopt.value @ trap.moments
l, = ax.plot(x / um, upot, label='pot')
# for k, mom in enumerate(trap.moments):
#     ax.plot(x / um, uopt.value[k] * mom, 'C3', alpha=0.2)

# curv, tilt, xComp, yComp, zComp, xCubic, vMesh, vGND
lab_params = (freq / MHz, -3.4, 6.9, 0.51, 13.92, 0, 0, 0)

_l = (0, ) * (len(lab_params) - 1)
volt_axial = calculate_voltage(lab_params[0], *_l)[:20]
volt_lab = calculate_voltage(*lab_params)[:20]

with np.printoptions(precision=4, linewidth=85, suppress=True):
    print('V')
    print(uopt.value)
    print('lab axial')
    print(volt_axial)
    print('lab full')
    print(volt_lab)  # with / 2.5, these are the voltages displayed on Ionizer

print(uopt.value.shape, volt_axial.shape)
# v_ax = reset_offset(v_ax)
pot_params(upot, l)

v_ax = volt_axial @ trap.moments
l, = ax.plot(x / um, v_ax, label='lab (axial only)')
pot_params(v_ax, l)

v_lab = volt_lab @ trap.moments
# v_lab = reset_offset(v_lab)

l, = ax.plot(x / um, v_lab, label='lab (full)')
pot_params(v_lab, l)

ax.legend()
plt.show()
