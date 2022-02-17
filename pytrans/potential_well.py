#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 01-2021 - Carmelo Mordini <carmelo> <cmordini@phys.ethz.ch>
"""
Hardcode for Ca+

"""

import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from .conversion import freq_to_curv, curv_to_freq

from typing import List

import logging
logger = logging.getLogger(__name__)


class GridPotential(ABC):

    @abstractmethod
    def potential(self, x: npt.ArrayLike) -> npt.ArrayLike:
        pass

    @abstractmethod
    def roi(self, x: npt.ArrayLike) -> npt.ArrayLike:
        pass

    @abstractmethod
    def weight(self, x: npt.ArrayLike) -> npt.ArrayLike:
        pass


class PotentialWell(GridPotential):
    """
    Just 1d for the moment, but here is where we'll generalize
    """

    def __init__(self, x0, axial, depth, offset=0, mass='Ca', charge=1, cutoff=1):

        self.x0 = x0
        self.depth = depth
        self.offset = offset
        self._cutoff = cutoff
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

    @property
    def cutoff(self):
        return self._cutoff * self.depth

    def __add__(self, other):
        if isinstance(other, PotentialWell):
            return MultiplePotentialWell([self, other])
        elif isinstance(other, MultiplePotentialWell):
            return MultiplePotentialWell(other.wells + [self])

    def potential(self, x):
        v = 0.5 * self.curvature * (x - self.x0)**2
        return v.clip(0, self.cutoff) + self.offset

    def gaussian_potential(self, x):
        if self.depth == 0:
            return np.zeros_like(x) + self.offset

        return self.depth - self.depth * np.exp(-(x - self.x0)**2 / 2 / self.sigma**2) + self.offset

    def roi(self, x):
        _roi = self.potential(x) - self.offset < self.cutoff  # change to logical array
        return _roi.astype(bool)

    def weight(self, x):
        return np.exp(-(x - self.x0)**2 / 2 / self.sigma**2)


class MultiplePotentialWell(GridPotential):

    def __init__(self, wells: List[PotentialWell]):
        self.wells = wells

    @property
    def axial(self):
        return [w._axial for w in self.wells]

    @property
    def curvature(self):
        return [w._curvature for w in self.wells]

    def potential(self, x):
        return np.stack([w.potential(x) for w in self.wells], axis=0).sum(0)

    def gaussian_potential(self, x):
        return np.stack([w.gaussian_potential(x) for w in self.wells], axis=0).sum(0)

    def roi(self, x):
        return np.stack([w.roi(x) for w in self.wells], axis=0).sum(0).astype(bool)

    def weight(self, x):
        return np.stack([w.weight(x) for w in self.wells], axis=0).sum(0)


class QuarticPotentialWell(GridPotential):
    """ Ref: Home, Steane, Electrode Configurations for Fast Separation of Trapped Ions
        https://arxiv.org/abs/quant-ph/0411102
    """

    def __init__(self, x0, alpha, beta, depth, offset=0, mass='Ca', charge=1, cutoff=4):

        self.x0 = x0
        self.alpha = alpha
        self.beta = beta
        self.depth = depth
        self.offset = offset
        self._cutoff = cutoff
        self._mass = ion_masses[mass] if isinstance(mass, str) else mass
        self._charge = charge * elementary_charge

    @property
    def cutoff(self):
        return self._cutoff * self.depth

    def potential(self, x):
        x1 = (x - self.x0)
        v = self.alpha * x1**2 + self.beta * x1**4
        return v.clip(v.min(), self.cutoff) + self.offset

    def roi(self, x):
        _roi = self.potential(x) - self.offset < self.cutoff  # change to logical array
        return _roi.astype(bool)

    def weight(self, x):
        sigma2 = 2 * abs(self.alpha) / self.beta  # 2 x the well separation at alpha < 0
        return np.exp(-(x - self.x0)**2 / 2 / sigma2)
