#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 01-2021 - Carmelo Mordini <cmordini> <cmordini@phys.ethz.ch>

"""
Plot segtrap electrode moments
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from scipy.io import loadmat

moments_path = Path(__file__).resolve().parent / 'DanielTrapMomentsTransport.mat'

fig, (ax0, ax1) = plt.subplots(2, 1)

data = loadmat(moments_path, struct_as_record=False)['DATA'][0][0]


def electrode_names(x):
    return ('DCC' + ('c' + str(x) if x < 15 else 'a' + str(x - 15)))


x = data.transport_axis.ravel()

for n in range(0, 30):
    ax = ax0 if n < 15 else ax1
    v = data.electrode[0, n].moments[:, 0]
    ax.plot(x, v, label=electrode_names(n))
fig.legend()
plt.show()
