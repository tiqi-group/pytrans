#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 01-2021 - Carmelo Mordini <carmelo> <cmordini@phys.ethz.ch>

import numpy as np
from .data import dc, pseudo
from ..abstract_trap import AbstractTrap

import logging
logger = logging.getLogger(__name__)


class TestTrap(AbstractTrap):
    """Simple model of a surface trap usin analytical moments
    """
    _electrodes = "E1 E2 E3 E4 E5 E6".split()
    v_rf = 30  # 30 volt
    omega_rf = 2 * np.pi * 30e6  # 30 MHz

    def __init__(self, use_electrodes='all'):

        pass

    def dc_potentials(self, x, y, z):
        return np.stack([getattr(dc, name)(x, y, z) for name in self.electrodes], axis=0)

    def pseudo_potential(self, x, y, z):
        return self.v_rf**2 / self.omega_rf**2 * pseudo.ps0(x, y, z)

    # Let's leave them not implemented to keep the example minimal
    def dc_gradients(self, x, y, z):
        raise NotImplementedError

    def dc_hessians(self, x, y, z):
        raise NotImplementedError

    def pseudo_gradient(self, x, y, z):
        raise NotImplementedError

    def pseudo_hessian(self, x, y, z):
        raise NotImplementedError


if __name__ == '__main__':

    class Trap(AbstractTrap):
        n_dc = 10
        pass
    trap = Trap()
