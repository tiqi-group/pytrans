#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 01-2021 - Carmelo Mordini <carmelo> <cmordini@phys.ethz.ch>
"""
Hardcode for Ca+

"""

import numpy as np
from .conversion import curv_to_freq, freq_to_curv, C

import logging
logger = logging.getLogger(__name__)


def get_voltage_params(axial, split, tilt, freq_pseudo):
    if not freq_pseudo:
        return np.ones((len(axial), 3, 3)) * np.nan
    tilt = tilt * np.pi / 180
    # assert -np.pi / 2 < tilt <= np.pi / 2, "Tilt angle must be within (-90, 90] degrees"
    v_ax = freq_to_curv(axial)
    v_ps = freq_to_curv(freq_pseudo)
    a = v_ps - v_ax / 2
    nu0 = curv_to_freq(a - freq_to_curv(split / 2))
    lamb = C * nu0 * split
    # nu1, nu2 = curv_to_freq(a + lamb) * 1e-6, curv_to_freq(a - lamb) * 1e-6
    # print(f"Transverse mode freqs: {nu1:.3f}, {nu2:.3f} MHz (split: {nu1 - nu2:.3f})")

    v_split = 2 * lamb * np.cos(tilt)**2 - lamb
    v_tilt = np.sign(tilt) * np.sqrt(lamb**2 - v_split**2)

    return np.asarray([v_ax, v_split, v_tilt, 0, 0, 0]) / C / 1e12


def get_hessian(axial, split, tilt, freq_pseudo):

    v_ax, v_split, v_tilt = get_voltage_params(axial, split, tilt, freq_pseudo)[:3] * C * 1e12
    v_ps = freq_to_curv(freq_pseudo)
    a = v_ps - v_ax / 2
    
    # TODO This is brutal. There should be a way to vectorize it but I'm lazy
    target_hessian = np.stack([
        [[_v_ax, 0, 0],
         [0, _a + _v_split, _v_tilt],
         [0, _v_tilt, _a - _v_split]] for _v_ax, _a, _v_split, _v_tilt in zip(v_ax, a, v_split, v_tilt)
    ])
    return target_hessian


def get_hessian_dc(axial, split):
    # force theta = 45, split is approximate
    v_ax = freq_to_curv(axial)
    b = C * split * 5.5e6
    target_hessian = np.stack([
        [[_v_ax, 0, 0],
         [0, -_v_ax / 2, _b],
         [0, _b, -_v_ax / 2]] for _v_ax, _b, in zip(v_ax, b)
    ])
    return target_hessian


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
        return - self.depth[sample] * np.exp(-(x - self.x0[sample])**2 / 2 / self.sigma[sample]**2)
