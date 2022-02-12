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
    __required_attributes = ['n_dc']

    def __new__(cls):
        print("Creating a new trap")
        for name in cls.__required_attributes:
            if not hasattr(cls, name):
                raise TypeError(f"Can't instantiate abstract class {cls.__name__} with abstract attributes {','.join(cls.__required_attributes)}")

        return super().__new__(cls)

    @abstractmethod
    def dc_potentials(self, x, y, z):
        """
        Returns:
            out: array_like, shape: (n_dc,) + x.shape
        """
        pass

    @abstractmethod
    def dc_gradients(self, x, y, z):
        """
        Returns:
            out: array_like, shape: (n_dc, 3) + x.shape
        """
        pass

    @abstractmethod
    def dc_hessians(self, x, y, z):
        """
        Returns:
            out: array_like, shape: (n_dc, 3, 3) + x.shape
        """
        pass

    @abstractmethod
    def pseudo_potential(self, x, y, z):
        """Pseudopotential from RF

        Returns:
            out: array_like, shape: x.shape
        """
        pass

    @abstractmethod
    def pseudo_gradient(self, x, y, z):
        """Pseudopotential gradient

        Returns:
            out: array_like, shape: (3,) + x.shape
        """
        pass

    @abstractmethod
    def pseudo_hessian(self, x, y, z):
        """Pseudopotential curvatures

        Returns:
            out: array_like, shape: (3, 3) + x.shape
        """
        pass

    def potential(self, voltages, x, y, z):
        return np.einsum('i,i...', voltages, self.dc_potentials(x, y, z)) + self.pseudo_potential(x, y, z)

    def gradient(self, voltages, x, y, z):
        return np.einsum('i,i...', voltages, self.dc_gradients(x, y, z)) + self.pseudo_gradient(x, y, z)

    def hessian(self, voltages, x, y, z):
        return np.einsum('i,i...', voltages, self.dc_hessians(x, y, z)) + self.pseudo_hessian(x, y, z)


if __name__ == '__main__':

    class Trap(AbstractTrap):
        n_dc = 10
        pass
    trap = Trap()
