#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 01-2021 - Carmelo Mordini <carmelo> <cmordini@phys.ethz.ch>
"""
Module docstring

"""

import numpy as np

from pytrans.trap_model.cryo import CryoTrap as Trap
from pytrans.potential_well import PotentialWell
from pytrans.solver import Solver

from pytrans.constants import um, MHz

x0 = np.linspace(0, 250 * um, 256)
depth = 0.1
axial = 1.3 * MHz
split = -2 * MHz
tilt = -3 * MHz

trap = Trap()
wells = [
    PotentialWell(x0, depth, axial, split, tilt, freq_pseudo=trap.freq_pseudo, scale_roi=1),
]
solver = Solver(trap, wells)


voltages = solver.solver()
