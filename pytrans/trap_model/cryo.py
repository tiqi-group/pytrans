#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 01-2021 - Carmelo Mordini <carmelo> <cmordini@phys.ethz.ch>

"""
Cryo trap model

"""

import numpy as np
from pathlib import Path
from pytrans.units import um
from .abstract_trap import AbstractTrap

import logging

logger = logging.getLogger(__name__)


class CryoTrap(AbstractTrap):
    """
    Cryo trap docstring
    """

    data_path = Path('/home/carmelo/ETH/pytrans/moments_data/cryo/csv')
    num_electrodes = 10  # they come in pairs

    def __init__(self):
        super().__init__()

    def load_trap_axis_potential_data(self):
        logger.info('Loading cryo trap data')
        self.potentials = []
        for j in range(self.num_electrodes):
            filepath = self.data_path / f'Axial2QubitTrap_xline_DC{j + 1}.csv'
            x, v = np.loadtxt(filepath, comments='%', delimiter=',', unpack=True)
            self.potentials.append(v)

        self.transport_axis = -x * um  # flip comsol reference frame and make it SI
        self.potentials = np.stack(self.potentials, axis=1)  # shape = (len(x), num_electrodes)
