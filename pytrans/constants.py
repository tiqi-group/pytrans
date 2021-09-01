#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 01-2021 - Carmelo Mordini <carmelo> <cmordini@phys.ethz.ch>

"""
Unit definitions, all in SI
"""

from scipy.constants import atomic_mass, elementary_charge, epsilon_0  # noqa

ion_masses = {
    'Ca': 39.962591 * atomic_mass,
    'Be': 9.012 * atomic_mass
}
um = 1e-6
us = 1e-6
ns = 1e-9
MHz = 1e6
kHz = 1e3
meV = 1e-3
