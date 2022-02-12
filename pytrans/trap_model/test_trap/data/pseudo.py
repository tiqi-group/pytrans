#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 02/2022
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>


from pytrans.conversion import freq_to_curv
from pytrans.constants import ion_masses, elementary_charge as _q

_m = ion_masses['Ca']

# Let's say the trap gives 5 MHz radial curvature
# when driven with an RF of 1 V and 1 MHz
_bare_rf_curv = freq_to_curv(5e6, _m, _q)
_rf_null_z0 = 50e-6


def ps0(x, y, z):
    return 0.5 * _bare_rf_curv * (y**2 + (z - _rf_null_z0)**2)
