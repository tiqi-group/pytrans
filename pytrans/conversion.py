#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 06/2021
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

'''
Module docstring
'''

import numpy as np
from .constants import mass_Ca, atomic_mass, elementary_charge, MHz, um

# def freq_to_curv(freq, mass=mass_Ca, charge=1):
#     return ((2 * np.pi * freq)**2 * atomic_mass_unit * mass /
#             (2 * electron_charge * charge))


# def curv_to_freq(curv, mass=mass_Ca, charge=1):
#     return (np.sqrt(2 * charge * electron_charge * curv / mass /
#                     atomic_mass_unit) / 2 / np.pi)

mass = mass_Ca * atomic_mass
C = mass / elementary_charge * (2 * np.pi)**2
E0 = C * MHz**2 * um  # E field that shifts x by 1 um


def curv_to_freq(curv):
    return np.sign(curv) * np.sqrt(np.abs(curv) / C)


def freq_to_curv(freq):
    return C * freq**2
