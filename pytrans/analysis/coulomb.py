#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 05/2022
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>


import numpy as np
from scipy.constants import epsilon_0

kappa = 1 / 4 / np.pi / epsilon_0


def coulomb_pot(x1, x2):
    return kappa / (x2 - x1)


def coulomb_grad(x1, x2):
    return kappa / (x2 - x1)**2 * np.asarray([1, -1])


def coulomb_hess(x1, x2):
    return 2 * kappa / (x2 - x1)**3 * np.asarray([[1, -1], [-1, 1]])
