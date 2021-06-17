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

axial = 1e6
split = (6.7 - 3.15) * 1e6
tilt = 45

bare_params = (1.9, -3, -2, 0, 0, 0)  # 2.1, 3, 3)
params = get_voltage_params(axial, split, tilt, trap.freq_pseudo)

print(params)
print(bare_params)

voltages = calculate_voltage(*params)[:20]
bvoltages = calculate_voltage(*bare_params)[:20]

fig, (axes1, axes2) = plot3d_make_layout(2)
print('\nmeasured')
analyse_pot(voltages, np.asarray((0, 0, trap.z0)), *args, axes=axes1)
print('\nbare params')
analyse_pot(bvoltages, np.asarray((0, 0, trap.z0)), *args, axes=axes2)

axes1[3].set_title('From measurements')
axes2[3].set_title('From bare params')
plt.show()
