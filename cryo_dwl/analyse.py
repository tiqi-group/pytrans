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
from calculate_voltage import calculate_voltage
from pytrans.trap_model.cryo import CryoTrap

trap = CryoTrap()

args = (
    trap.electrode_indices,
    trap.Vrf,
    trap.Omega_rf
)

params = (1.9, -5, 3, 0.3, -1.8)
voltages = calculate_voltage(*params)[:20]

analyse_pot(voltages, np.asarray((0, 0, trap.z0)), *args, axes=None)
plt.show()
