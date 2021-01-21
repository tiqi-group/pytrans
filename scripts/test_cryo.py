#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 01-2021 - Carmelo Mordini <carmelo> <cmordini@phys.ethz.ch>

"""
Module docstring

"""

from pytrans.trap_model.cryo import CryoTrap

trap = CryoTrap()
print(trap.data_path)
trap.load_trap_axis_potential_data()

print(trap.transport_axis.shape)
print(trap.potentials.shape)

import matplotlib.pyplot as plt

plt.plot(trap.transport_axis, trap.potentials)
plt.show()
