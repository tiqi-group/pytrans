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


# Hydrogen
H2 = Ion("H2", mass_amu=2.016, unit_charge=1)

# Beryllium
Be9 = Ion("Be9", mass_amu=9.012, unit_charge=1)

# Magnesium
Mg24 = Ion("Mg24", mass_amu=23.985041, unit_charge=1)

# Calcium
Ca40 = Ion("Ca40", mass_amu=39.962591, unit_charge=1)
Ca42 = Ion("Ca42", mass_amu=41.958618, unit_charge=1)
Ca43 = Ion("Ca43", mass_amu=42.958766, unit_charge=1)
Ca44 = Ion("Ca44", mass_amu=43.955482, unit_charge=1)
Ca46 = Ion("Ca46", mass_amu=45.953688, unit_charge=1)
Ca48 = Ion("Ca48", mass_amu=47.952534, unit_charge=1)

# Strontium
Sr88 = Ion("Sr88", mass_amu=87.9056122, unit_charge=1)

# Barium
Ba137 = Ion("Ba137", mass_amu=136.9058274, unit_charge=1)
Ba138 = Ion("Ba138", mass_amu=137.9052472, unit_charge=1)

# Ytterbium
Yb171 = Ion("Yb171", mass_amu=170.936323, unit_charge=1)
Yb174 = Ion("Yb174", mass_amu=173.9388621, unit_charge=1)

# Mercury
Hg199 = Ion("Hg199", mass_amu=198.9682799, unit_charge=1)

# Aluminum
Al27 = Ion("Al27", mass_amu=26.98153841, unit_charge=1)
