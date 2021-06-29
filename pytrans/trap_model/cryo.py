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
from pytrans.utils.timer import timer

from .moments_data.cryo.analytic import potentialsDC, gradientsDC, hessiansDC, pseudoPotential

import logging
logger = logging.getLogger(__name__)

filename_basis = Path(__file__).resolve().parent / 'moments_data/cryo/vbasis_x0.npy'
vb0 = np.load(filename_basis)


class CryoTrap(AbstractTrap):
    """
    Cryo trap docstring
    """

    num_electrodes = 20
    default_V = 5
    min_V = -10
    max_V = 10
    z0 = 5.16792281e-05
    Vrf = 40
    Omega_rf = 2 * np.pi * 34e6
    freq_pseudo = 5.6075e6  # this actually depends on the other two

    def __init__(self, x=None):
        super().__init__()
        self.transport_axis = self.x = np.arange(-1000, 1005, 5) * 1e-6 if x is None else x  # nice to have an alias
        self.electrode_indices = list(range(1, self.num_electrodes + 1))
        self.load_trap_axis_potential_data()

    @timer
    def load_trap_axis_potential_data(self):
        self.moments = np.stack([
            getattr(potentialsDC, f"E{index}")(self.transport_axis, 0, self.z0) for index in self.electrode_indices
        ], axis=0)  # (num_electrodes, len(x))
        self.gradients = np.stack([
            getattr(gradientsDC, f"E{index}")(self.transport_axis, 0, self.z0) for index in self.electrode_indices
        ], axis=0)  # (num_electrodes, 3, len(x))
        self.hessians = np.stack([
            getattr(hessiansDC, f"E{index}")(self.transport_axis, 0, self.z0).reshape(9, -1) for index in self.electrode_indices
        ], axis=0)  # (num_electrodes, 9, len(x))

        self.pseudo_potential = pseudoPotential.ps0(self.transport_axis, 0, self.z0, self.Vrf, self.Omega_rf)  # (len(x),)
        self.pseudo_gradient = pseudoPotential.ps1(self.transport_axis, 0, self.z0, self.Vrf, self.Omega_rf)  # (3, len(x))
        self.pseudo_hessian = pseudoPotential.ps2(self.transport_axis, 0, self.z0, self.Vrf, self.Omega_rf).reshape(9, -1)  # (9, len(x))

    def calculate_voltage(self, axial, split, tilt, x_comp=0, y_comp=0, z_comp=0):  # , xCubic, vMesh, vGND, xyTilt=0, xzTilt=0):
        # Array of voltages. 20 electrodes
        # voltages = (axial, split, tilt, x_comp, y_comp, z_comp) @ vb0
        v0 = np.asarray([axial, split, tilt]) * 1e-6
        v0 = np.sign(v0) * v0**2
        v0 = np.r_[v0, x_comp, y_comp, z_comp]
        voltages = v0 @ vb0
        return voltages


class CryoTrapFunctions(AbstractTrap):
    """
    Cryo trap docstring
    """

    num_electrodes = 20
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
        self.transport_axis = np.linspace(-1000, 1000, 2001) * 1e-6  # dummy, you don't actually need it with the functions
        self.moments = [partial(self._electrode_potential, index=index) for index in self.electrode_indices]
        self.gradients = [partial(self._electrode_gradient, index=index) for index in self.electrode_indices]
        self.hessians = [partial(self._electrode_hessian, index=index) for index in self.electrode_indices]

    def _electrode_potential(self, x, index):
        return getattr(potentialsDC, f"E{index}")(x, 0, self.z0)

    def _electrode_gradient(self, x, index):
        return getattr(gradientsDC, f"E{index}")(x, 0, self.z0)

    def _electrode_hessian(self, x, index):
        return getattr(hessiansDC, f"E{index}")(x, 0, self.z0)

    def eval_moments(self, x):
        return np.stack([m(x) for m in self.moments], axis=0)  # (num_electrodes, len(x))

    def eval_gradient(self, x):
        return np.stack([e(x) for e in self.gradients], axis=0)  # (num_electrodes, 3)

    def eval_hessian(self, x):
        return np.stack([h(x) for h in self.hessians], axis=0)  # (num_electrodes, 3, 3)

    def pseudo_potential(self, x):
        return pseudoPotential.ps0(x, 0, self.z0, self.Vrf, self.Omega_rf)

    def pseudo_gradient(self, x):
        return pseudoPotential.ps1(x, 0, self.z0, self.Vrf, self.Omega_rf)

    def pseudo_hessian(self, x):
        return pseudoPotential.ps2(x, 0, self.z0, self.Vrf, self.Omega_rf)

    def calculate_voltage(self, axial, split, tilt, x_comp=0, y_comp=0, z_comp=0):  # , xCubic, vMesh, vGND, xyTilt=0, xzTilt=0):
        # Array of voltages. 20 electrodes
        # voltages = (axial, split, tilt, x_comp, y_comp, z_comp) @ vb0
        v0 = np.asarray([axial, split, tilt]) * 1e-6
        v0 = np.sign(v0) * v0**2
        v0 = np.r_[v0, x_comp, y_comp, z_comp]
        voltages = v0 @ vb0
        return voltages

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
