#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 02/2022
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

import numpy as np
from .trap import TestTrap, trap


def test_dc_potentials(trap: TestTrap):
    x = np.linspace(-200, 200, 100) * 1e-6
    dc = trap.dc_potentials(x, 0, 0)
    assert dc.shape == (trap.n_dc,) + x.shape
