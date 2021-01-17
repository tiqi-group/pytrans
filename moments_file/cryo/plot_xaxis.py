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


fig, ax = plt.subplots()

for electrode_n in range(1, 11):
    fpath = Path('.') / 'csv' / f'Axial2QubitTrap_xline_DC{electrode_n}x0-0.csv'

    x, v = np.loadtxt(fpath, comments='%', delimiter=',', unpack=True)

    ax.plot(x, v, label=electrode_n)
ax.legend()
plt.show()
