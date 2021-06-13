#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 06/2021
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

'''
Module docstring
'''
import matplotlib.pyplot as plt
from pytrans.trap_model.cryo import CryoTrap
from pytrans.utils.cryo_plotting import plot_electrodes

trap = CryoTrap()
print(trap.electrode_indices)

x = trap.transport_axis

for j in range(trap.num_electrodes):
    m1 = trap.moments[j](x)
    l, = plt.plot(x * 1e6, m1, label=f"Ele{trap.electrode_indices[j]}")

plot_electrodes(plt.gca())
plt.legend()
plt.show()
