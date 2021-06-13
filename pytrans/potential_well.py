#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 01-2021 - Carmelo Mordini <carmelo> <cmordini@phys.ethz.ch>
"""
Hardcode for Ca+

"""

import numpy as np
from pytrans.constants import mass_Ca, atomic_mass, elementary_charge

import logging
logger = logging.getLogger(__name__)


# def freq_to_curv(freq, mass=mass_Ca, charge=1):
#     return ((2 * np.pi * freq)**2 * atomic_mass_unit * mass /
#             (2 * electron_charge * charge))


# def curv_to_freq(curv, mass=mass_Ca, charge=1):
#     return (np.sqrt(2 * charge * electron_charge * curv / mass /
#                     atomic_mass_unit) / 2 / np.pi)

mass = mass_Ca * atomic_mass
C = mass / elementary_charge * (2 * np.pi)**2


def curv_to_freq(curv):
    return np.sign(curv) * np.sqrt(np.abs(curv) / C)


def freq_to_curv(freq):
    return C * freq**2


def get_hessian(axial, split, tilt, freq_pseudo):
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

    # TODO This is brutal. There should be a way to vectorize it but I'm lazy
    target_hessian = np.stack([
        [[_v_ax, 0, 0],
         [0, _a + _v_split, _v_tilt],
         [0, _v_tilt, _a - _v_split]] for _v_ax, _a, _v_split, _v_tilt in zip(v_ax, a, v_split, v_tilt)
    ])
    return target_hessian


class PotentialWell:
    """
    Just 1d for the moment, but here is where we'll generalize
    """

    def __init__(self, x0, depth, axial, split, tilt, freq_pseudo=5.6075e6):
        """
        A moving potential well
        First index of all the arrays is reserved for the timesteps
        Stored values (x0, axial etc.): (time,)
        potential: (time) + x.shape
        hessian: (time, 3, 3)

        Args:
        moments
        """

        self.x0, self.depth, self.axial, self.split, self.tilt = \
            np.broadcast_arrays(np.atleast_1d(x0), depth, axial, split, tilt)
        self.timesteps = len(self.x0)
        self.freq_pseudo = freq_pseudo

        self.sigma = np.sqrt(self.depth / C) / self.axial
        self.curv = freq_to_curv(self.axial)

        self.hessian = get_hessian(self.axial, self.split, self.tilt, self.freq_pseudo)

    def roi(self, x, step=0):
        return abs(x - self.x0[step]) < 4 * self.sigma[step]  # change to logical array

    def weight(self, x, step=0):
        return np.exp(-(x - self.x0[step])**2 / 2 / self.sigma[step]**2)

    def potential(self, x, step=0, clip=True):
        pot = 0.5 * self.curv[step] * (x - self.x0[step])**2 - self.depth[step]
        if clip:
            pot = pot.clip(None, 0)
        return pot

    def gaussian_potential(self, x, step=0):
        return - self.depth[step] * np.exp(-(x - self.x0[step])**2 / 2 / self.sigma[step]**2)
