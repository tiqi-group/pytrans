#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 01-2021 - Carmelo Mordini <cmordini> <cmordini@phys.ethz.ch>
"""
Load cryo trap electrode moments from Karan's csv files
"""

import numpy as np


def load_1d_data(fpath):
    x, v = np.loadtxt(fpath, comments='%', delimiter=',', unpack=True)
    return x, v


def load_2d_data(fpath, shape):
    x, y, v = np.loadtxt(fpath, comments='%', delimiter=',', unpack=True)
    x = x.reshape(shape)
    y = y.reshape(shape)
    v = v.reshape(shape)
    return x, y, v
