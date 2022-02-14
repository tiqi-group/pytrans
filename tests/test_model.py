#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 02/2022
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

import numpy as np
from .trap import TestTrap, trap


def test_use_electrodes():
    trap = TestTrap(use_electrodes=['E2', 'E5'])
    assert trap.electrodes == ['E2', 'E5']


def test_el_index(trap: TestTrap):
    assert trap.el_index(['E1', 'E4']) == [0, 3]
    assert trap.el_index('E2') == 1


def test_dc_potentials(trap: TestTrap):
    x = np.linspace(-200, 200, 100) * 1e-6
    dc = trap.dc_potentials(x, 0, 0)
    assert dc.shape == (trap.n_electrodes,) + x.shape
