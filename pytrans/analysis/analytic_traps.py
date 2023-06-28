#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 06/2023
# MDH

import numpy as np
from scipy.constants import pi, elementary_charge, epsilon_0

from pytrans.ions import Ion, atomic_mass
from pytrans.conversion import curv_to_freq

from itertools import permutations

kappa = elementary_charge / 4 / pi / epsilon_0


class AnalyticTrap:

    def __add__(self, other):
        if not isinstance(other, AnalyticTrap):
            raise TypeError('Both objects must be an instance of AnalyticTrap')
        return CombinedTrap(self, other)

    def _ravel_coords(self, *args):
        args = np.broadcast_arrays(*args)
        shape = args[0].shape
        args = list(map(np.ravel, args))
        X = np.stack(args, axis=1).astype(float)
        return shape, X


class CombinedTrap(AnalyticTrap):
    def __init__(self, trap_a: AnalyticTrap, trap_b: AnalyticTrap):
        self._trap_a = trap_a
        self._trap_b = trap_b

    def potential(self, voltages, x, y, z, mass_amu, pseudo=True):
        return self._trap_a.potential(voltages, x, y, z, mass_amu, pseudo) + self._trap_b.potential(voltages, x, y, z, mass_amu, pseudo)

    def gradient(self, voltages, x, y, z, mass_amu, pseudo=True):
        return self._trap_a.gradient(voltages, x, y, z, mass_amu, pseudo) + self._trap_b.gradient(voltages, x, y, z, mass_amu, pseudo)

    def hessian(self, voltages, x, y, z, mass_amu, pseudo=True):
        return self._trap_a.hessian(voltages, x, y, z, mass_amu, pseudo) + self._trap_b.hessian(voltages, x, y, z, mass_amu, pseudo)


class HarmonicTrap(AnalyticTrap):

    def __init__(self, fx, fy, fz, ion: Ion, field=[0, 0, 0],
                 tilt_xy=0, tilt_xz=0, tilt_yz=0):
        wx2 = np.sign(fx) * (2 * pi * fx)**2
        wy2 = np.sign(fy) * (2 * pi * fy)**2
        wz2 = np.sign(fz) * (2 * pi * fz)**2
        c_x = ion.mass / ion.charge * wx2
        c_dc = ion.mass / ion.charge * (wy2 - wz2) / 2
        m_c_rf = ion.mass**2 / ion.charge * (wx2 + wy2 + wz2) / 2

        self.rf_null_coords = (None, 0, 0)

        self._H_dc = np.asarray([
            [c_x, tilt_xy, tilt_xz],
            [tilt_xy, c_dc - c_x / 2, tilt_yz],
            [tilt_xz, tilt_yz, -c_dc - c_x / 2]
        ])

        self._m_H_rf = np.asarray([
            [0, 0, 0],
            [0, m_c_rf, 0],
            [0, 0, m_c_rf]
        ])

        self._E = np.asarray(field)

    def _H(self, mass_amu):
        mass = atomic_mass * np.atleast_1d(mass_amu).reshape(-1, 1, 1)
        return self._H_dc.reshape(1, 3, 3) + self._m_H_rf / mass

    def potential(self, voltages, x, y, z, mass_amu, pseudo=True):
        shape, X = self._ravel_coords(x, y, z)
        H = self._H(mass_amu)
        pot = 0.5 * np.einsum('...i,...ij,...j', X, H, X) + \
            np.einsum('j,...j', self._E, X)
        return pot.reshape(shape)

    def gradient(self, voltages, x, y, z, mass_amu, pseudo=True):
        shape, X = self._ravel_coords(x, y, z)
        H = self._H(mass_amu)
        grad = np.einsum('...ij,...j', H, X) + self._E.reshape(1, -1)
        return grad.reshape(shape + (3,))

    def hessian(self, voltages, x, y, z, mass_amu, pseudo=True):
        shape, X = self._ravel_coords(x, y, z)
        H = self._H(mass_amu)
        return H.reshape(shape + (3, 3))

    def trap_frequencies(self, ion: Ion):
        curv = np.diag(self._H(ion.mass_amu))
        return curv_to_freq(curv, ion=ion)


class CubicTrap(AnalyticTrap):
    def __init__(self, a_cubic: float):
        self._C = np.zeros((1, 3, 3, 3))
        self._C[0, 0, 0, 0] = a_cubic
        for p in set(permutations((0, 1, 1))):
            self._C[(0,) + p] = - a_cubic / 2
        for p in set(permutations((0, 2, 2))):
            self._C[(0,) + p] = - a_cubic / 2

    def potential(self, voltages, x, y, z, mass_amu, pseudo=True):
        shape, X = self._ravel_coords(x, y, z)
        pot = (1 / 6.) * np.einsum('...abc,...a,...b,...c', self._C, X, X, X)
        return pot.reshape(shape)

    def gradient(self, voltages, x, y, z, mass_amu, pseudo=True):
        shape, X = self._ravel_coords(x, y, z)
        grad = 0.5 * np.einsum('...iab,...a,...b', self._C, X, X)
        return grad.reshape(shape + (3,))

    def hessian(self, voltages, x, y, z, mass_amu, pseudo=True):
        shape, X = self._ravel_coords(x, y, z)
        H = np.einsum('...ija,...a', self._C, X)
        return H.reshape(shape + (3, 3))


class QuarticTrap(AnalyticTrap):

    def __init__(self, a_quartic: float):
        self._C = np.zeros((1,) + (3,) * 4)
        self._C[0, 0, 0, 0, 0] = a_quartic
        for p in set(permutations((0, 0, 1, 1))):
            self._C[(0,) + p] = - a_quartic / 2
        for p in set(permutations((0, 0, 2, 2))):
            self._C[(0,) + p] = - a_quartic / 2

    def potential(self, voltages, x, y, z, mass_amu, pseudo=True):
        shape, X = self._ravel_coords(x, y, z)
        pot = (1 / 24.) * np.einsum('...abcd,...a,...b,...c,...d', self._C, X, X, X, X)
        return pot.reshape(shape)

    def gradient(self, voltages, x, y, z, mass_amu, pseudo=True):
        shape, X = self._ravel_coords(x, y, z)
        grad = (1 / 6.) * np.einsum('...jabc,...a,...b,...c', self._C, X, X, X)
        return grad.reshape(shape + (3,))

    def hessian(self, voltages, x, y, z, mass_amu, pseudo=True):
        shape, X = self._ravel_coords(x, y, z)
        H = 0.5 * np.einsum('...ijab,...a,...b', self._C, X, X)
        return H.reshape(shape + (3, 3))
