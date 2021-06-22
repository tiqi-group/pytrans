#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 01-2021 - Carmelo Mordini <carmelo> <cmordini@phys.ethz.ch>
"""
Module docstring

"""
import numpy as np
import matplotlib.pyplot as plt

from pytrans.utils.cryo_analysis import analyse_pot, plot3d_make_layout
from calculate_voltage import calculate_voltage
from pytrans.trap_model.cryo import CryoTrap

from pytrans.potential_well import get_voltage_params

trap = CryoTrap()

args = (
    trap.electrode_indices,
    trap.Vrf,
    trap.Omega_rf
)

axial = 2e6
split = -1.3e6
tilt = 30

params = get_voltage_params(axial, split, tilt, trap.freq_pseudo)

print('Calc:', params)

voltages = calculate_voltage(*params)[:20]

print('Voltages:', voltages)
fig, axes1 = plot3d_make_layout(1)
print('\nmeasured')
analyse_pot(voltages, np.asarray((0, 0, trap.z0)), *args, axes=axes1)

axes1[3].set_title('From measurements')
plt.show()
