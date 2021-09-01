#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 01-2021 - Carmelo Mordini <carmelo> <cmordini@phys.ethz.ch>
"""
Cryo trap model

"""

import numpy as np
from pathlib import Path
from math import factorial

from ..abstract_trap import AbstractTrap
from pytrans.utils.timer import timer
from pytrans.utils import indexing as uix

from .data.analytic import potentialsDC, gradientsDC, hessiansDC, third_order_axial_DC, fourth_order_axial_DC, pseudoPotential
from .data.calculate_voltage import calculate_voltage as _calculate_voltage
from .fastino_wf_gen import generate_waveform as _generate_waveform

import logging
logger = logging.getLogger(__name__)

cache_filename = Path(__file__).resolve().parent / 'data/cached_data.npz'


class CryoTrap(AbstractTrap):
    """
    Cryo trap docstring
    """

    _num_electrodes = 20
    default_V = 5
    min_V = -10
    max_V = 10
    z0 = 5.16792281e-05
    Vrf = 40
    Omega_rf = 2 * np.pi * 34e6
    freq_pseudo = 5.6075e6  # this actually depends on the other two
    dc_gain = 2.5

    _d_names = [[""]] + [s.split() for s in [
        "x y z",
        "xx xy xz yy yz zz",
        "xxx",
        "xxxx"
    ]]

    _d_map = uix.populate_map(_d_names)

    def __init__(self, x=None, selected_electrodes=None, use_cache=True):
        super().__init__()
        self._x = np.arange(-1000, 1005, 5) * 1e-6 if x is None else x
        self.selected_electrodes = slice(0, self._num_electrodes) if selected_electrodes is None else selected_electrodes
        self._electrode_indices = np.asarray(range(1, self._num_electrodes + 1))
        self._electrode_x = np.asarray([(n - 6) * 125e-6 for n in range(1, 11)] * 2)
        self.load_trap_axis_potential_data(use_cache)

    @property
    def x(self):
        return self._x

    @property
    def transport_axis(self):
        return self._x

    @property
    def electrode_indices(self):
        return self._electrode_indices[self.selected_electrodes]

    @property
    def electrode_x(self):
        return self._electrode_x[self.selected_electrodes]

    @property
    def num_electrodes(self):
        return len(self.electrode_indices)

    @timer
    def load_trap_axis_potential_data(self, use_cache):
        if use_cache and cache_filename.exists():
            A = np.load(cache_filename)
            if np.array_equal(A['x'], self.x):
                self._dc_potential = A['dc']
                self._pseudo_potential = A['pseudo']
                print("Load from cache")
                return
        self._load_data()

    def _load_data(self):
        data = [
            [
                getattr(potentialsDC, f"E{index}")(self.x, 0, self.z0)[np.newaxis, :],           # (1, len(x))
                getattr(gradientsDC, f"E{index}")(self.x, 0, self.z0),                           # (3, len(x))
                getattr(hessiansDC, f"E{index}")(self.x, 0, self.z0)[np.triu_indices(3)],        # (6, len(x))
                getattr(third_order_axial_DC, f"E{index}")(self.x, 0, self.z0)[np.newaxis, :],   # (1, len(x))
                getattr(fourth_order_axial_DC, f"E{index}")(self.x, 0, self.z0)[np.newaxis, :],  # (1, len(x))
            ]
            for index in self._electrode_indices
        ]
        self._dc_potential = np.stack([np.concatenate(d, axis=0) for d in data], axis=0)  # (num_electrodes, 12, len(x))

        data = [
            pseudoPotential.ps0(self.x, 0, self.z0, self.Vrf, self.Omega_rf)[np.newaxis, :],        # (1, len(x))
            pseudoPotential.ps1(self.x, 0, self.z0, self.Vrf, self.Omega_rf),                       # (3, len(x))
            pseudoPotential.ps2(self.x, 0, self.z0, self.Vrf, self.Omega_rf)[np.triu_indices(3)],   # (6, len(x))
        ]
        self._pseudo_potential = np.concatenate(data, axis=0)  # (10, len(x))
        print("Save to cache")
        np.savez(cache_filename, x=self.x, dc=self._dc_potential, pseudo=self._pseudo_potential)

    def dc_potential(self, derivatives):
        derivative_indices = uix.get_derivative(derivatives, self._d_map)
        return self._dc_potential[self.selected_electrodes][:, derivative_indices, :]

    def pseudo_potential(self, derivatives):
        derivative_indices = uix.get_derivative(derivatives, self._d_map)
        return self._pseudo_potential[derivative_indices, :]

    def potential(self, voltages):
        assert len(voltages) == self._num_electrodes, "Need all voltages here"
        return voltages @ self._dc_potential[:, 0, :] + self._pseudo_potential[0, :]

    @property
    def moments(self):
        return np.squeeze(self.dc_potential(derivatives=0))  # (num_electrodes, len(x))

    @property
    def taylor_moments(self):
        f = np.asarray([1 / factorial(j) for j in range(5)]).reshape(1, -1, 1)
        derivatives = [''.join(['x'] * j) for j in range(5)]
        taylor = f * self.dc_potential(derivatives)
        return taylor

    def from_static_params(self, axial, tilt, x_comp=0, y_comp=0, z_comp=0, center=6):  # , xCubic, vMesh, vGND, xyTilt=0, xzTilt=0):
        voltages = _calculate_voltage(axial * 1e-6, tilt * 1e-6, x_comp, y_comp, z_comp, center)
        potential = voltages @ self._dc_potential[:, 0, :]
        return voltages, potential

    def generate_waveform(self, voltages, index, description='', generated=True, uid=None,
                          waveform_filename=None, verbose=False,
                          monitor_values=None):
        assert len(voltages.shape) == 2, "Voltages must be 2d (time, electrodes)"
        full_voltages = np.zeros((voltages.shape[0], self._num_electrodes + 6))
        full_voltages[:, self.selected_electrodes] = voltages
        if monitor_values is not None:
            full_voltages[:, -1] = monitor_values
        return _generate_waveform(full_voltages / self.dc_gain, index, description, generated, uid,
                                  waveform_filename, verbose)
