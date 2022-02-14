#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 02/2022
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

import pytest
import numpy as np
import matplotlib.pyplot as plt
from .trap import TestTrap, trap

pytest.skip(allow_module_level=True)


def test_plot(trap: TestTrap):
    fig, (ax_x, ax_y, ax0, ax1) = plt.subplots(1, 4, figsize=(18, 4))

    z0 = 50e-6
    x = np.linspace(-200, 200, 100) * 1e-6
    y = np.linspace(-20, 20, 30) * 1e-6
    z = np.linspace(-20, 20, 30) * 1e-6 + z0

    moments = trap.dc_potentials(x, 0, z0)
    for name, v in zip(trap.electrodes, moments):
        ax_x.plot(x * 1e6, v, label=name)

    moments = trap.dc_potentials(0, y, z0)
    for name, v in zip(trap.electrodes, moments):
        ax_y.plot(y * 1e6, v, label=name)

    ax_y.legend()

    Y, Z = np.meshgrid(y, z)

    try:
        sel = trap.el_index(['E2', 'E5'])
    except ValueError:
        sel = list(range(trap.n_electrodes))
    vv = trap.dc_potentials(0, Y, Z)[sel].sum(0)
    ax0.contour(Y * 1e6, Z * 1e6, vv, 50)
    ax0.set_aspect(1)
    ax0.set_title(" + ".join([trap.electrodes[j] for j in sel]))

    ps = trap.pseudo_potential(0, Y, Z)
    ax1.contour(Y * 1e6, Z * 1e6, ps, 50)
    ax1.set_aspect(1)

    plt.show()


def test_plot2():
    trap2 = TestTrap(use_electrodes=['E1'])
    test_plot(trap2)
