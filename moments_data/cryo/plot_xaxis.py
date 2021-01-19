#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 01-2021 - Carmelo Mordini <cmordini> <cmordini@phys.ethz.ch>

"""
Plot cryo trap electrode moments
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# plt.rcParams['toolbar'] = 'toolmanager'


fig, ax = plt.subplots()

for electrode_n in range(1, 11):
    fpath = Path('.') / 'csv' / f'Axial2QubitTrap_xline_DC{electrode_n}x0-0.csv'

    x, v = np.loadtxt(fpath, comments='%', delimiter=',', unpack=True)

    ax.plot(x, v, label=electrode_n)
ax.set(xlabel='x [um]', ylabel='xline_DC moment [V]')
ax.legend()

# fig.savefig('xline_moments.png')

# for the 2d stuff: see Plot2dData.m in Karan's folder
# for electrode_n in range(1, 11):
# electrode_n = 1
# fpath = Path('.') / 'csv' / f'Axial2QubitTrap_xplane_DC{electrode_n}.csv'
#
# x, y, v = np.loadtxt(fpath, comments='%', delimiter=',', unpack=True)
#
# fig, ax = plt.subplots()
# ax.plot(x,)

plt.show()
