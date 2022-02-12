#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 02/2022
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

import pytest
import numpy as np
import matplotlib.pyplot as plt
from .trap import TestTrap, trap


# @pytest.mark.skip
def test_plot(trap: TestTrap):
    fig, (ax, ax1) = plt.subplots(1, 2, figsize=(12, 4))

    z0 = 50e-6
    x = np.linspace(-200, 200, 100) * 1e-6
    y = np.linspace(-20, 20, 30) * 1e-6
    z = np.linspace(-20, 20, 30) * 1e-6 + z0

    moments = trap.dc_potentials(x, 0, z0)
    for name, v in zip(trap.electrodes, moments):
        ax.plot(x * 1e6, v, label=name)

    Y, Z = np.meshgrid(y, z)
    ps = trap.pseudo_potential(0, Y, Z)
    c0 = ax1.contour(Y * 1e6, Z * 1e6, ps, 50)
    ax1.set_aspect(1)

    plt.show()
