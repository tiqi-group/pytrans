#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 01-2021 - Carmelo Mordini <carmelo> <cmordini@phys.ethz.ch>

"""
Unit definitions, all in SI
"""

from scipy.constants import atomic_mass, elementary_charge


class Ion:
    def __init__(self, name, mass_amu, unit_charge=1):
        self.mass_amu = mass_amu
        self.mass = mass_amu * atomic_mass
        self.unit_charge = unit_charge
        self.charge = unit_charge * elementary_charge
        self.__name = name

    def __repr__(self):
        return self.__name


Ca40 = Ion("Ca40", mass_amu=39.962591, unit_charge=1)
Be9 = Ion("Be9", mass_amu=9.012, unit_charge=1)
Mg24 = Ion("Mg24", mass_amu=23.985041, unit_charge=1)
