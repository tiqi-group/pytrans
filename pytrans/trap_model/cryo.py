#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 01-2021 - Carmelo Mordini <carmelo> <cmordini@phys.ethz.ch>
"""
Cryo trap model

"""

import numpy as np
from pathlib import Path
from functools import partial
from .abstract_trap import AbstractTrap

from .moments_data.cryo.analytic import potentialsDC

import logging
logger = logging.getLogger(__name__)


class CryoTrap(AbstractTrap):
    """
    Cryo trap docstring
    """

    data_path = Path(__file__) / "moments_data/cryo/csv"
    num_electrodes = 20  # they come in pairs
    default_V = 5
    min_V = -10
    max_V = 10
    z0 = 5.16792281e-05
    Vrf = 40
    Omega_rf = 2 * np.pi * 34e6
    freq_pseudo = 5.6075e6  # this actually depends on the other two

    def __init__(self):
        super().__init__()
        self.load_trap_axis_potential_data()

    def load_trap_axis_potential_data(self):
        self.electrode_indices = list(range(1, self.num_electrodes + 1))
        self.transport_axis = np.linspace(-3000, 3000, 1000) * 1e-6  # dummy, you don't actually need it with the functions
        self.moments = [partial(self._electrode_potential, index=index) for index in self.electrode_indices]

    def _electrode_potential(self, x, index):
        return getattr(potentialsDC, f"E{index}")(x, 0, self.z0)

    def eval_moments(self, x):
        return np.stack([m(x) for m in self.moments], axis=0)  # (num_ele * len(x))

    # def _load_trap_axis_potential_data_from_comsol(self):
    #     logger.info('Loading cryo trap data')
    #     self.moments = []
    #     for j in range(10):
    #         filepath = self.data_path / f'Axial2QubitTrap_xline_DC{j + 1}.csv'
    #         x, v = np.loadtxt(filepath, comments='%', delimiter=',', unpack=True)
    #         self.moments.append(v)
    #     self.moments.extend(self.moments)
    #     self.transport_axis = -x * um  # flip comsol reference frame and make it SI
    #     self.moments = np.stack(self.moments, axis=0)  # shape = (num_electrodes, len(x))
