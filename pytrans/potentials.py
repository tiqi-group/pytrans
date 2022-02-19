#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 02/2022
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>


import numpy as np

from .ions import Ion
from .conversion import freq_to_curv


def gaussian_1d(x, x0, sigma):
    return np.exp(-(x - x0)**2 / 2 / sigma**2)


def quadratic_potential_1d(x, x0, freq, offset, ion: Ion):
    curv = freq_to_curv(freq, ion=ion)
    return 0.5 * curv * (x - x0)**2 + offset
