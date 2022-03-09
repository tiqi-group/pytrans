#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 01-2021 - Carmelo Mordini <carmelo> <cmordini@phys.ethz.ch>

import numpy as np
from abc import ABC, abstractmethod

import logging
logger = logging.getLogger(__name__)


class AbstractTrap(ABC):
    """This class does nothing, but it will force the implementation
    of some properties and methods in trap models that are used in the Objectives

    Required:
    - functions for 3d dc potentials and derivatives
    - functions for 3d RF pseudopotential and derivatives

    Optional?:
    - num_electrodes
    - electrode - channel mapping, for waveforms
    - electrode_constrains (min, max, slew rate)
    - filters?
    """
    __required_attributes = ['_electrodes', 'v_rf', 'omega_rf']

    def __new__(cls, *args, **kwargs):
        for name in cls.__required_attributes:
            if not hasattr(cls, name):
                raise TypeError(f"Can't instantiate abstract class {cls.__name__} with abstract attributes {','.join(cls.__required_attributes)}")
        cls._all_electrodes = cls._electrodes
        return super().__new__(cls)

    def __init__(self, electrodes=None):
        if electrodes is not None:
            for name in electrodes:
                if name not in self._all_electrodes:
                    raise AttributeError(f"Trap {self.__class__.__name__} has no electrode {name}")
            self._electrodes = electrodes

    @property
    def electrodes(self):
        return self._electrodes

    @property
    def n_electrodes(self):
        """Number of active electrodes
        """
        return len(self.electrodes)

    @property
    def electrode_all_indices(self):
        return self.electrode_to_index(self.electrodes, in_all=True)

    def electrode_to_index(self, names, in_all=False):
        el_list = self._all_electrodes if in_all else self.electrodes
        if isinstance(names, str):
            return el_list.index(names)
        return [el_list.index(n) for n in names]

    @abstractmethod
    def dc_potentials(self, x, y, z):
        """
        Returns:
            out: array_like, shape: (n_dc,) + x.shape
        """
        raise NotImplementedError

    @abstractmethod
    def dc_gradients(self, x, y, z):
        """
        Returns:
            out: array_like, shape: (n_dc, 3) + x.shape
        """
        raise NotImplementedError

    @abstractmethod
    def dc_hessians(self, x, y, z):
        """
        Returns:
            out: array_like, shape: (n_dc, 3, 3) + x.shape
        """
        raise NotImplementedError

    @abstractmethod
    def pseudo_potential(self, x, y, z):
        """Pseudopotential from RF

        Returns:
            out: array_like, shape: x.shape
        """
        raise NotImplementedError

    @abstractmethod
    def pseudo_gradient(self, x, y, z):
        """Pseudopotential gradient

        Returns:
            out: array_like, shape: (3,) + x.shape
        """
        raise NotImplementedError

    @abstractmethod
    def pseudo_hessian(self, x, y, z):
        """Pseudopotential curvatures

        Returns:
            out: array_like, shape: (3, 3) + x.shape
        """
        raise NotImplementedError

    def potential(self, voltages, x, y, z, pseudo=True):
        u = np.tensordot(voltages, self.dc_potentials(x, y, z), axes=1)
        if pseudo:
            u += self.pseudo_potential(x, y, z)
        return u

    def gradient(self, voltages, x, y, z, pseudo=True):
        u = np.tensordot(voltages, self.dc_gradients(x, y, z), axes=1)
        if pseudo:
            u += self.pseudo_gradient(x, y, z)
        return u

    def hessian(self, voltages, x, y, z, pseudo=True):
        u = np.tensordot(voltages, self.dc_hessians(x, y, z), axes=1)
        if pseudo:
            u += self.pseudo_hessian(x, y, z)
        return u


if __name__ == '__main__':

    class Trap(AbstractTrap):
        _electrodes = ["E1", "E2"]
        v_rf = 1.
        omega_rf = 2 * np.pi * 1e6

    trap = Trap()
