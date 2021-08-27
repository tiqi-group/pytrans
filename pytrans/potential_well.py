#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 01-2021 - Carmelo Mordini <carmelo> <cmordini@phys.ethz.ch>
"""
Hardcode for Ca+

"""

import numpy as np
from .conversion import freq_to_curv, curv_to_freq
from .constants import ion_masses, elementary_charge

import logging
logger = logging.getLogger(__name__)


class PotentialWell:
    """
    Just 1d for the moment, but here is where we'll generalize
    """

    def __init__(self, x0, axial, depth, offset=0, mass='Ca', charge=1):

        self.x0 = x0
        self.offset = offset
        self.depth = depth
        self._axial = axial
        self._mass = ion_masses[mass] if isinstance(mass, str) else mass
        self._charge = charge * elementary_charge
        self._curvature = freq_to_curv(self._axial, self._mass, self._charge)

    @property
    def axial(self):
        return self._axial

    @axial.setter
    def axial(self, value):
        self._axial = value
        self._curvature = freq_to_curv(self._axial, self._mass, self._charge)

    @property
    def curvature(self):
        return self._curvature

    @curvature.setter
    def curvature(self, value):
        self._curvature = value
        self._axial = curv_to_freq(self._curvature, self._mass, self._charge)

    @property
    def sigma(self):
        return np.sqrt(self.depth / self.curvature)

    def potential(self, x):
        return 0.5 * self.curvature * (x - self.x0)**2 + self.offset

    def gaussian_potential(self, x):
        if self.depth == 0:
            return np.zeros_like(x) + self.offset

        return self.depth - self.depth * np.exp(-(x - self.x0)**2 / 2 / self.sigma**2) + self.offset

    def roi(self, x):
        return self.potential(x) < 2 * self.depth  # change to logical array

    def weight(self, x):
        return np.exp(-(x - self.x0)**2 / 2 / self.sigma**2)
