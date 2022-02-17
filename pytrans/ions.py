#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 01-2021 - Carmelo Mordini <carmelo> <cmordini@phys.ethz.ch>

"""
Unit definitions, all in SI
"""

from scipy.constants import atomic_mass, elementary_charge


class Ion:
    def __init__(self, name, mass, charge):
        self.mass = mass
        self.charge = charge
        self.__name = name

    def __repr__(self):
        return self.__name


Ca40 = Ion("Ca40", 39.962591 * atomic_mass, elementary_charge)
Be9 = Ion("Be9", 9.012 * atomic_mass, elementary_charge)
