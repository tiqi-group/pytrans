#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 01-2021 - Carmelo Mordini <carmelo> <cmordini@phys.ethz.ch>
"""
Hardcode for Ca+

"""

import numpy as np
from .conversion import freq_to_curv, get_hessian, get_hessian_dc, C

import logging
logger = logging.getLogger(__name__)


class PotentialWell:
    """
    Just 1d for the moment, but here is where we'll generalize
    """

    def __init__(self, x0, depth, axial, split, tilt, freq_pseudo=None, scale_roi=2):
        """
        A moving potential well
        First index of all the arrays is reserved for the number of samples (time)
        Stored values (x0, axial etc.): (time,)
        potential: (time) + x.shape
        hessian: (time, 3, 3)

        Args:
        moments
        """

        self.x0, self.depth, self.axial, self.split, self.tilt = \
            np.broadcast_arrays(np.atleast_1d(x0), depth, axial, split, tilt)
        self.samples = len(self.x0)
        self.freq_pseudo = freq_pseudo

        self.sigma = np.sqrt(self.depth / C) / self.axial
        self.curv = freq_to_curv(self.axial)

        self.hessian = get_hessian(self.axial, self.split, self.tilt, self.freq_pseudo)
        self.hessian_dc = get_hessian_dc(self.axial, self.split)
        self.scale_roi = scale_roi

    def roi(self, x, sample=0):
        return self.potential(x, sample) < self.scale_roi**2 * self.depth[sample]  # change to logical array

    def weight(self, x, sample=0):
        return np.exp(-(x - self.x0[sample])**2 / 2 / (self.scale_roi * self.sigma[sample])**2)

    def potential(self, x, sample=0):
        return 0.5 * self.curv[sample] * (x - self.x0[sample])**2

    def gaussian_potential(self, x, sample=0):
        if self.depth[sample] == 0:
            return np.zeros_like(x)
        return - self.depth[sample] * np.exp(-(x - self.x0[sample])**2 / 2 / self.sigma[sample]**2)
